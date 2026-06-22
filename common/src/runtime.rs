//! PMIx/UCX runtime initialization.
//!
//! Provides a unified `OsUContext` that wraps communication layers:
//! - **PMIx** — bootstrap, rank/size discovery, address exchange
//! - **UCX** — point-to-point tag matching, endpoints, progress engine
//!
//! UCC support will be added later for collective benchmarks.
//!
//! Pattern derived from gups-rs which uses the same stack for the GUPS benchmark.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let ctx = OsUContext::init().expect("init");
//! let partner = 1 - ctx.rank();
//! // ... use ctx.endpoint(partner), ctx.worker(), etc.
//! drop(ctx); // finalizes UCX → PMIx
//! ```

use std::ffi::CString;

use ucx_sys::context;
use ucx_sys::ep;
use ucx_sys::worker;
use ucx_sys::worker::RemoteWorkerAddress;
use ucx_sys::RequestParamBuilder;

use pmix::{commit, fence, get_value, init, put_value, Context, GLOBAL, PmixValueBuilder, RANK_WILDCARD};

// ── PMIx key names for data exchange ──

const PMIX_KEY_UCX_ADDR: &str = "osu.ucx.addr";

// ── Public API ──

/// Unified OSU benchmark context wrapping PMIx + UCX.
///
/// Created once at startup and shared across all benchmarks.
/// Drop order (UCX worker → UCX context → PMIx) is handled automatically.
pub struct OsUContext {
    rank: usize,
    size: usize,
    /// UCX endpoints — one per rank (including self).
    endpoints: Vec<ep::Ep>,
    /// UCX worker for progress / tag recv.
    worker: worker::Worker,
    /// UCX context (kept alive for worker lifetime).
    _ucx_context: context::Context,
    /// PMIx context — kept alive; drop calls PMIx_Finalize.
    _pmix_ctx: Context,
}

impl OsUContext {
    /// Create a new unified context.
    ///
    /// This function:
    /// 1. Initializes PMIx for rank/namespace discovery
    /// 2. Queries job size
    /// 3. Initializes UCX context with Tag features
    /// 4. Creates UCX worker and packs address
    /// 5. Publishes worker address via PMIx_Put + Commit + Fence
    /// 6. Retrieves peer addresses via PMIx_Get
    /// 7. Creates UCX endpoints to all peers
    /// 8. Progresses and flushes endpoints
    pub fn init() -> Self {
        // 1. Initialize PMIx — gets our rank
        let pmix_ctx = init(None).expect("PMIx init");
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

        // 3. Initialize UCX context — Tag-only for OSU benchmarks
        let features = context::Flags::Tag;
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
        fence(&my_proc, None).expect("PMIx_Fence");

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
                // Self endpoint
                let own_remote_addr = RemoteWorkerAddress::new(own_addr_bytes.clone());
                let ep_params = ep::ParamsBuilder::new().address(&own_remote_addr).build();
                let ep = worker.create_ep(&ep_params).expect("Self EP create");
                endpoints.push(ep);
                continue;
            }
            let remote_addr = RemoteWorkerAddress::new(peer_addrs[peer].clone());
            let ep_params = ep::ParamsBuilder::new().address(&remote_addr).build();
            let ep = worker.create_ep(&ep_params).expect("EP create for peer");
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

        OsUContext {
            rank,
            size,
            endpoints,
            worker,
            _ucx_context: ucx_context,
            _pmix_ctx: pmix_ctx,
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

    /// Progress the UCX worker (drain pending operations).
    pub fn progress(&self) {
        loop {
            if !self.worker.progress() {
                break;
            }
        }
    }

    /// Simple barrier using UCX tag matching (all-to-all handshake).
    pub fn barrier(&self) {
        let rank = self.rank;
        let size = self.size;
        let worker = &self.worker;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

        const BARRIER_TAG: u64 = 0xBEEFCAFE;
        const TAG_MASK: u64 = u64::MAX;

        if size <= 1 {
            return;
        }

        // Send barrier message to all peers
        let msg = [rank as u8];
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer).tag_send(&msg, BARRIER_TAG, &tag_param)
                    .expect("barrier send");
            }
        }

        // Receive barrier messages from all peers
        let mut recv_buf = [0u8; 1];
        for peer in 0..size {
            if peer != rank {
                let req = worker.tag_recv(&mut recv_buf, BARRIER_TAG, TAG_MASK, &tag_param)
                    .expect("barrier recv")
                    .expect("barrier recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }
    }

    /// Allreduce a u64 value using UCX tag matching (ring algorithm).
    pub fn allreduce_u64(&self, value: u64) -> u64 {
        let rank = self.rank;
        let size = self.size;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

        const REDUCE_TAG: u64 = 0xDEADBEEF;
        const TAG_MASK: u64 = u64::MAX;

        if size <= 1 {
            return value;
        }

        // Simple all-gather then sum approach
        let mut gathered = vec![0u64; size];
        gathered[rank] = value;

        // Send our value to all peers
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer).tag_send(&value.to_le_bytes(), REDUCE_TAG, &tag_param)
                    .expect("reduce send");
            }
        }

        // Receive from all peers
        let mut recv_buf = [0u8; 8];
        for peer in 0..size {
            if peer != rank {
                let req = self.worker.tag_recv(&mut recv_buf, REDUCE_TAG, TAG_MASK, &tag_param)
                    .expect("reduce recv")
                    .expect("reduce recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                gathered[peer] = u64::from_le_bytes(recv_buf);
            }
        }

        gathered.iter().sum()
    }
}

/// Flush an endpoint by flushing the worker.
fn flush_ep_blocking(worker: &worker::Worker, _ep: &ep::Ep, param: &ucx_sys::RequestParam) {
    // Worker.flush() flushes all AM/RMA on this worker.
    let req = worker.flush(param);
    if let Ok(Some(r)) = req {
        while !r.check_finished().unwrap_or(false) {
            loop {
                if !worker.progress() {
                    break;
                }
            }
        }
    }
}
