//! OsUContext struct and initialization logic.
//!
//! `needless_range_loop` is allowed because `for peer in 0..size` is the
//! idiomatic MPI-style rank iteration pattern used throughout this module.

#![allow(clippy::needless_range_loop)]

use std::ffi::CString;

use ucx_sys::context;
use ucx_sys::ep;
use ucx_sys::memh;
use ucx_sys::rma::RemoteKey;
use ucx_sys::worker;
use ucx_sys::worker::RemoteWorkerAddress;

use pmix::{
    Context, GLOBAL, PmixValueBuilder, RANK_WILDCARD, commit, fence, get_value,
    info_with_string_key, init, put_value,
};

use crate::runtime::constants::*;
use crate::runtime::helpers::flush_ep_blocking;
use crate::runtime::ucc_oob::init_ucc;

/// Backend in use for collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// UCC library — native collective operations.
    Ucc,
    /// UCX tag-matching fallback for collectives.
    Ucx,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Ucc => write!(f, "UCC"),
            Backend::Ucx => write!(f, "UCX (tag-matching fallback)"),
        }
    }
}

/// How the backend was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// User forced UCC with --ucc.
    ForcedUcc,
    /// User disabled UCC with --no-ucc.
    ForcedUcx,
    /// Auto-detected — tried UCC, succeeded.
    AutoUcc,
    /// Auto-detected — UCC unavailable, fell back to UCX.
    AutoUcx,
}

impl std::fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendSelection::ForcedUcc => write!(f, "forced (--ucc)"),
            BackendSelection::ForcedUcx => write!(f, "forced (--no-ucc)"),
            BackendSelection::AutoUcc => write!(f, "auto-detected (UCC available)"),
            BackendSelection::AutoUcx => {
                write!(f, "auto-detected (UCC unavailable, UCX fallback)")
            }
        }
    }
}

/// Resolve the PMIx server URI from the local URI file for standalone clients.
///
/// When running under `prterun`, PMIx_Init finds the server automatically via the
/// environment variables that prterun sets (PMIX_RANK, PMIX_SERVER_URI61, etc.).
/// The C library reads these internally — no explicit URI needed.
///
/// Only resolve the URI file when NOT under prterun (standalone mode) and a
/// system server daemon is running. This avoids connecting to stale daemons.
///
/// Lookup: URI file at `/run/user/{uid}/prte/uri`
fn resolve_pmix_server_uri() -> Option<String> {
    // When running under prterun, let PMIx_Init discover the server via env vars.
    // Do NOT pass pmix.srvr.uri to PMIx_Init — that key is for PMIx_Tool_Init,
    // and passing it to PMIx_Init causes ErrUnreach with OpenPMIX 6.1.0.
    if std::env::var("PMIX_RANK").is_ok() {
        return None;
    }

    // Standalone mode: try to connect to a running system server via URI file
    // Use getuid() not getpid() — the URI lives under /run/user/{uid}/
    let uid = unsafe { libc::getuid() };
    let uri_path = format!("/run/user/{}/prte/uri", uid);
    if let Ok(content) = std::fs::read_to_string(&uri_path) {
        let uri = content.lines().next()?.trim().to_string();
        if !uri.is_empty() {
            return Some(uri);
        }
    }
    None
}

/// Unified OSU benchmark context wrapping PMIx + UCX + UCC.
///
/// Created once at startup and shared across all benchmarks.
/// Drop order (UCC team → UCX worker → UCX context → PMIx) is handled automatically.
pub struct OsUContext {
    pub rank: usize,
    pub size: usize,
    /// UCX endpoints — one per rank (including self).
    pub endpoints: Vec<ep::Ep>,
    /// UCX worker for progress / tag recv.
    pub worker: worker::Worker,
    /// UCX context (kept alive for worker lifetime).
    pub _ucx_context: context::Context,
    /// PMIx context — kept alive; drop calls PMIx_Finalize.
    pub _pmix_ctx: Context,
    /// UCC team for collective operations (Some when UCC init succeeded).
    pub ucc_team: Option<ucc::team::UccTeam>,
    /// Which backend is active for collectives.
    pub backend: Backend,
    /// How the backend was selected.
    pub backend_selection: BackendSelection,
    /// Remote keys for RMA operations (one per peer, None for self).
    pub remote_rkeys: Vec<Option<RemoteKey>>,
    /// Remote memory addresses for RMA target buffers (one per rank).
    pub remote_mem_addrs: Vec<u64>,
    /// Own registered memory handle (for RMA targets).
    pub _memh: Option<memh::MemHandle>,
}

impl OsUContext {
    /// Create a new unified context.
    ///
    /// `ucc_backend` controls UCC initialization:
    /// - `Some(true)`  — force UCC, panic if it fails
    /// - `Some(false)` — skip UCC entirely, UCX fallback only
    /// - `None`        — auto-detect (try UCC, fall back to UCX on failure)
    ///
    /// `rma_target` is an optional buffer to register for RMA one-sided operations.
    /// When provided, the buffer is registered with UCX, the rkey is exchanged via
    /// PMIx, and remote rkeys are unpacked for all peers. This enables one-sided
    /// benchmarks (osu_put_latency, osu_get_latency, osu_acc_latency).
    pub fn init_with_rma(ucc_backend: Option<bool>, rma_target: Option<&mut [u8]>) -> Self {
        // 1. Initialize PMIx — gets our rank
        // When under prterun: init(None) lets the C library discover the server
        // via env vars (PMIX_RANK, PMIX_SERVER_URI61, etc.) that prterun sets.
        // When standalone: resolve_pmix_server_uri() tries the system server URI file.
        let pmix_info =
            resolve_pmix_server_uri().map(|uri| info_with_string_key("pmix.srvr.uri", &uri));
        let pmix_ctx = init(pmix_info).expect("PMIx init");
        let rank = pmix_ctx.get_rank() as usize;
        let my_proc = pmix_ctx.get_proc();

        // 2. Query job size
        let wc_proc = pmix_ctx
            .proc_with_nspace(RANK_WILDCARD)
            .expect("wildcard_proc");

        let size = get_value(&wc_proc, pmix::JOB_SIZE, None)
            .ok()
            .map(|v| v.uint32() as usize)
            .or_else(|| {
                get_value(&wc_proc, "PMIX_JOB_SIZE\0".as_bytes(), None)
                    .ok()
                    .map(|v| v.uint32() as usize)
            })
            .or_else(|| std::env::var("PMIX_SIZE").ok().and_then(|s| s.parse().ok()))
            .unwrap_or_else(|| {
                eprintln!("[osu] Warning: Could not determine job size, defaulting to 2");
                2
            });
        eprintln!("[osu] PMIx rank={}, size={}", rank, size);

        // 3. Initialize UCX context
        // Tag for point-to-point benchmarks, RMA + ExportedMemH only for one-sided benchmarks.
        // On machines without RDMA (no InfiniBand/RoCE), requesting Rma forces UCX to look
        // for put/compare-and-swap transports that don't exist, causing UCS_ERR_UNREACHABLE
        // on cross-process endpoints. Only request RMA when actually needed.
        let mut features = context::Flags::Tag;
        if rma_target.is_some() {
            features |= context::Flags::Rma | context::Flags::ExportedMemH;
        }
        let ctx_params = context::ParamsBuilder::new()
            .features(features)
            .estimated_num_eps(size - 1)
            .estimated_num_ppn(2)
            .build();
        let config = context::Config::default();
        let ucx_context = context::Context::new(&config, &ctx_params).expect("UCX context init");
        drop(config);

        // 4. Create worker
        let wparams = worker::ParamsBuilder::new().build();
        let worker = ucx_context
            .worker_create(&wparams)
            .expect("UCX worker create");

        // 5. Pack own worker address
        let packed_addr = worker.pack_address().expect("Worker address pack");
        let own_addr_bytes = packed_addr.to_vec();

        // 6. Register RMA target memory (if provided)
        let (memh, rma_enabled) = if let Some(buf) = rma_target {
            let mut mem_params = memh::MemMapParamsBuilder::new();
            mem_params
                .address(buf.as_mut_ptr() as *mut std::ffi::c_void)
                .length(buf.len());
            let handle = memh::MemHandle::map(&ucx_context, &mut mem_params)
                .expect("RMA memory registration");
            (Some(handle), true)
        } else {
            (None, false)
        };

        // 7. Pack rkey for RMA target memory (if registered)
        let own_rkey_data: Option<Vec<u8>> = if let Some(ref handle) = memh {
            let packed_rkey = memh::pack_rkey(&ucx_context, handle).expect("Rkey pack");
            Some(packed_rkey.as_bytes().to_vec())
        } else {
            None
        };

        // 8. Get our memory address for RMA (if registered)
        let own_mem_addr: u64 = if let Some(ref handle) = memh {
            handle.query().expect("Memh query").address() as u64
        } else {
            0
        };

        // 9. Publish our address via PMIx_Put
        let addr_key = CString::new(PMIX_KEY_UCX_ADDR).unwrap();
        let mut addr_val = PmixValueBuilder::new()
            .byte_object(&own_addr_bytes)
            .expect("byte_object addr")
            .build()
            .expect("build addr");
        put_value(GLOBAL, &addr_key, &mut addr_val).expect("PMIx_Put addr");

        // 10. Publish RMA data (if applicable)
        if let Some(ref rkey_bytes) = own_rkey_data {
            let rkey_key = CString::new(PMIX_KEY_RKEY).unwrap();
            let mut rkey_val = PmixValueBuilder::new()
                .byte_object(rkey_bytes)
                .expect("byte_object rkey")
                .build()
                .expect("build rkey");
            put_value(GLOBAL, &rkey_key, &mut rkey_val).expect("PMIx_Put rkey");
        }
        if let Some(handle) = &memh {
            let mem_addr = handle.query().expect("Memh query").address() as u64;
            let mem_key = CString::new(PMIX_KEY_MEM_ADDR).unwrap();
            let mut mem_val = PmixValueBuilder::new()
                .uint64(mem_addr)
                .build()
                .expect("build mem_addr");
            put_value(GLOBAL, &mem_key, &mut mem_val).expect("PMIx_Put mem_addr");
        }

        // 11. Commit + Fence (barrier + data exchange)
        commit().expect("PMIx_Commit");
        fence(my_proc, None).expect("PMIx_Fence");

        // 12. Retrieve peer addresses via PMIx_Get
        let mut peer_addrs: Vec<Vec<u8>> = vec![Vec::new(); size];
        peer_addrs[rank] = own_addr_bytes.clone();

        // 13. Retrieve peer RMA data via PMIx_Get
        let mut peer_rkey_data: Vec<Vec<u8>> = vec![Vec::new(); size];
        let mut remote_mem_addrs = vec![0u64; size];
        remote_mem_addrs[rank] = own_mem_addr;

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let remote_proc = pmix_ctx
                .proc_with_nspace(peer as u32)
                .expect("proc_with_nspace");

            // Get worker address
            let addr_key_bytes = format!("{}{}", PMIX_KEY_UCX_ADDR, '\0');
            let addr_val =
                get_value(&remote_proc, addr_key_bytes.as_bytes(), None).expect("PMIx_Get addr");
            peer_addrs[peer] = addr_val.bytes_copy();

            // Get RMA data (if available)
            if rma_enabled {
                let rkey_key_bytes = format!("{}{}", PMIX_KEY_RKEY, '\0');
                let rkey_val = get_value(&remote_proc, rkey_key_bytes.as_bytes(), None)
                    .expect("PMIx_Get rkey");
                peer_rkey_data[peer] = rkey_val.bytes_copy();

                let mem_key_bytes = format!("{}{}", PMIX_KEY_MEM_ADDR, '\0');
                let mem_val = get_value(&remote_proc, mem_key_bytes.as_bytes(), None)
                    .expect("PMIx_Get mem_addr");
                remote_mem_addrs[peer] = mem_val.uint64();
            }
        }

        drop(packed_addr);

        // 14. Create UCX endpoints to each peer
        let mut endpoints = Vec::with_capacity(size);
        for peer in 0..size {
            if peer == rank {
                let own_remote_addr = RemoteWorkerAddress::new(own_addr_bytes.clone());
                let ep_params = ep::ParamsBuilder::new().address(&own_remote_addr).build();
                let ep = worker.create_ep(ep_params).expect("Self EP create");
                endpoints.push(ep);
                continue;
            }
            let remote_addr = RemoteWorkerAddress::new(peer_addrs[peer].clone());
            let ep_params = ep::ParamsBuilder::new().address(&remote_addr).build();
            let ep = worker.create_ep(ep_params).expect("EP create for peer");
            endpoints.push(ep);
        }

        // 15. Progress endpoint connections
        loop {
            if !worker.progress() {
                break;
            }
        }

        // 16. Unpack remote rkeys (if RMA is enabled)
        let mut remote_rkeys: Vec<Option<RemoteKey>> = (0..size).map(|_| None).collect();
        if rma_enabled {
            for peer in 0..size {
                if peer == rank {
                    continue;
                }
                let rkey = RemoteKey::unpack(&endpoints[peer], &peer_rkey_data[peer])
                    .expect("rkey unpack");
                remote_rkeys[peer] = Some(rkey);
            }
        }

        // 17. Flush all endpoints
        let flush_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
        for peer in 0..size {
            flush_ep_blocking(&worker, &endpoints[peer], &flush_param);
        }

        eprintln!("[osu] UCX ready (rank={}, size={})", rank, size);

        // 18. Initialize UCC library, context, and team (conditional)
        let ucc_team = match ucc_backend {
            Some(true) => {
                eprintln!("[osu] UCC backend: forced on (--ucc)");
                match init_ucc(&worker, &endpoints, rank, size) {
                    Some(team) => Some(team),
                    None => panic!(
                        "UCC initialization failed but --ucc was specified. \
                         Try --no-ucc to use UCX fallback, or install UCC library."
                    ),
                }
            }
            Some(false) => {
                eprintln!("[osu] UCC backend: disabled (--no-ucc), using UCX fallback only");
                None
            }
            None => {
                eprintln!("[osu] UCC backend: auto-detect (use --ucc or --no-ucc to override)");
                init_ucc(&worker, &endpoints, rank, size)
            }
        };

        // Determine backend and selection based on ucc_backend param and result
        let (backend, backend_selection) = match ucc_backend {
            Some(true) => {
                if ucc_team.is_some() {
                    (Backend::Ucc, BackendSelection::ForcedUcc)
                } else {
                    unreachable!("UCC init failed with --ucc — should have panicked")
                }
            }
            Some(false) => (Backend::Ucx, BackendSelection::ForcedUcx),
            None => {
                if ucc_team.is_some() {
                    (Backend::Ucc, BackendSelection::AutoUcc)
                } else {
                    (Backend::Ucx, BackendSelection::AutoUcx)
                }
            }
        };

        // Report final backend selection
        eprintln!("[osu] Backend: {} ({})", backend, backend_selection);

        OsUContext {
            rank,
            size,
            endpoints,
            worker,
            _ucx_context: ucx_context,
            _pmix_ctx: pmix_ctx,
            ucc_team,
            backend,
            backend_selection,
            remote_rkeys,
            remote_mem_addrs,
            _memh: memh,
        }
    }

    /// Create a new unified context (without RMA target registration).
    ///
    /// `ucc_backend` controls UCC initialization:
    /// - `Some(true)`  — force UCC, panic if it fails
    /// - `Some(false)` — skip UCC entirely, UCX fallback only
    /// - `None`        — auto-detect (try UCC, fall back to UCX on failure)
    pub fn init(ucc_backend: Option<bool>) -> Self {
        Self::init_with_rma(ucc_backend, None)
    }

    /// Get the rank of this process.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the total number of processes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the UCX endpoint to a specific rank.
    pub fn endpoint(&self, rank: usize) -> &ep::Ep {
        &self.endpoints[rank]
    }

    /// Get a reference to the UCX worker.
    pub fn worker(&self) -> &worker::Worker {
        &self.worker
    }

    /// Get a reference to the UCC team (if available).
    pub fn ucc_team(&self) -> Option<&ucc::team::UccTeam> {
        self.ucc_team.as_ref()
    }

    /// Progress the UCX worker (drain pending operations).
    pub fn progress(&self) {
        loop {
            if !self.worker.progress() {
                break;
            }
        }
    }

    /// Get the remote memory address for a peer (for RMA operations).
    pub fn remote_mem_addr(&self, peer: usize) -> u64 {
        self.remote_mem_addrs[peer]
    }

    /// Get the remote key for a peer (for RMA operations).
    pub fn remote_rkey(&self, peer: usize) -> Option<&RemoteKey> {
        self.remote_rkeys[peer].as_ref()
    }
}
