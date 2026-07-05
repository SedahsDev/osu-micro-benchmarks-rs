//! OsUContext struct and initialization logic.
//!
//! `needless_range_loop` is allowed because `for peer in 0..size` is the
//! idiomatic MPI-style rank iteration pattern used throughout this module.

#![allow(clippy::needless_range_loop)]

use std::ffi::CString;

use ucx_sys::RequestParamBuilder;
use ucx_sys::context;
use ucx_sys::ep;
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
            BackendSelection::AutoUcx => write!(f, "auto-detected (UCC unavailable, UCX fallback)"),
        }
    }
}

/// Resolve the PMIx server URI from environment or local file.
///
/// OpenPMIX 6.1.0 ignores `PMIX_SERVER_URI` set via `std::env::set_var`,
/// so we read the URI ourselves and pass it through `info_with_string_key`.
///
/// Lookup order:
/// 1. `PMIX_SERVER_URI` environment variable
/// 2. URI file at `/run/user/{uid}/prte/uri`
/// 3. `None` if neither is available (bare `PMIx_Init` as before)
#[allow(clippy::collapsible_if)]
fn resolve_pmix_server_uri() -> Option<String> {
    // 1. Check environment variable
    if let Ok(uri) = std::env::var("PMIX_SERVER_URI") {
        if !uri.is_empty() {
            return Some(uri);
        }
    }
    // 2. Read URI file from systemd runtime directory
    let uid = std::process::id();
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
}

impl OsUContext {
    /// Create a new unified context.
    ///
    /// `ucc_backend` controls UCC initialization:
    /// - `Some(true)`  — force UCC, panic if it fails
    /// - `Some(false)` — skip UCC entirely, UCX fallback only
    /// - `None`        — auto-detect (try UCC, fall back to UCX on failure)
    pub fn init(ucc_backend: Option<bool>) -> Self {
        // 1. Initialize PMIx — gets our rank
        // Pass server URI explicitly (OpenPMIX 6.1.0 ignores env vars for this)
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

        // 3. Initialize UCX context — Tag + RMA + AMO64 + ExportedMemH for full benchmark coverage
        let features = context::Flags::Tag
            | context::Flags::Rma
            | context::Flags::Amo64
            | context::Flags::ExportedMemH;
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

        // 6. Publish our address via PMIx_Put
        let addr_key = CString::new(PMIX_KEY_UCX_ADDR).unwrap();
        let mut addr_val = PmixValueBuilder::new()
            .byte_object(&own_addr_bytes)
            .expect("byte_object addr")
            .build()
            .expect("build addr");
        put_value(GLOBAL, &addr_key, &mut addr_val).expect("PMIx_Put addr");

        // 7. Commit + Fence (barrier + data exchange)
        commit().expect("PMIx_Commit");
        fence(my_proc, None).expect("PMIx_Fence");

        // 8. Retrieve peer addresses via PMIx_Get
        let mut peer_addrs: Vec<Vec<u8>> = vec![Vec::new(); size];
        peer_addrs[rank] = own_addr_bytes.clone();

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let remote_proc = pmix_ctx
                .proc_with_nspace(peer as u32)
                .expect("proc_with_nspace");

            let addr_key_bytes = format!("{}{}", PMIX_KEY_UCX_ADDR, '\0');
            let addr_val =
                get_value(&remote_proc, addr_key_bytes.as_bytes(), None).expect("PMIx_Get addr");
            peer_addrs[peer] = addr_val.bytes_copy();
        }

        drop(packed_addr);

        // 9. Create UCX endpoints to each peer
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

        // 10. Progress endpoint connections
        loop {
            if !worker.progress() {
                break;
            }
        }

        // 11. Flush all endpoints
        let flush_param = RequestParamBuilder::new().no_imm_cmpl().build();
        for peer in 0..size {
            flush_ep_blocking(&worker, &endpoints[peer], &flush_param);
        }

        eprintln!("[osu] UCX ready (rank={}, size={})", rank, size);

        // 12. Initialize UCC library, context, and team (conditional)
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
                    // Already panicked above, but handle for exhaustiveness
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
        }
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
}
