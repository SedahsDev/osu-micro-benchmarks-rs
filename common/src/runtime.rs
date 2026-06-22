//! PMIx/UCX/UCC runtime initialization.
//!
//! Provides a unified `OsUContext` that wraps communication layers:
//! - **PMIx** — bootstrap, rank/size discovery, address exchange
//! - **UCX** — point-to-point tag matching, endpoints, progress engine
//! - **UCC** — native collective operations built on top of UCX
//!
//! UCC is initialized with OOB (out-of-band) callbacks that use UCX tag
//! matching for the allgather needed during context/team creation.
//!
//! Pattern derived from gups-rs which uses the same stack for the GUPS benchmark.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let ctx = OsUContext::init().expect("init");
//! let partner = 1 - ctx.rank();
//! // ... use ctx.endpoint(partner), ctx.worker(), ctx.ucc_team(), etc.
//! drop(ctx); // finalizes UCC → UCX → PMIx
//! ```

use std::ffi::CString;
use std::os::raw::c_void;

use ucx_sys::context;
use ucx_sys::ep;
use ucx_sys::worker;
use ucx_sys::worker::RemoteWorkerAddress;
use ucx_sys::RequestParamBuilder;

use pmix::{commit, fence, get_value, init, put_value, Context, GLOBAL, PmixValueBuilder, RANK_WILDCARD};

// ── PMIx key names for data exchange ──

const PMIX_KEY_UCX_ADDR: &str = "osu.ucx.addr";

// ── UCC OOB tag ──
// Tag base for UCC out-of-band allgather. Each peer uses 0xCC0000 + peer_rank.
const UCC_OOB_TAG_BASE: u64 = 0xCC0000;
const UCC_OOB_TAG_MASK: u64 = u64::MAX;

// ── Public API ──

/// Unified OSU benchmark context wrapping PMIx + UCX + UCC.
///
/// Created once at startup and shared across all benchmarks.
/// Drop order (UCC team → UCX worker → UCX context → PMIx) is handled automatically.
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
    /// UCC team for collective operations (Some when UCC init succeeded).
    ucc_team: Option<ucc::team::UccTeam>,
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
    /// 9. Initializes UCC library + context + team (with OOB via UCX)
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

        // 12. Initialize UCC library, context, and team
        let ucc_team = init_ucc(&worker, &endpoints, rank, size);

        OsUContext {
            rank,
            size,
            endpoints,
            worker,
            _ucx_context: ucx_context,
            _pmix_ctx: pmix_ctx,
            ucc_team,
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

    /// Reduce a f64 value to find the minimum across all ranks (UCX fallback).
    pub fn allreduce_min_f64(&self, value: f64) -> f64 {
        let bits = value.to_bits();
        let summed = self.allreduce_u64(bits);
        // We need proper min reduction — use allgather + local min
        let rank = self.rank;
        let size = self.size;
        let mut gathered = vec![0u64; size];
        gathered[rank] = bits;
        // Reuse the allgather pattern from allreduce_u64
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const MIN_TAG: u64 = 0xDEAD0001;
        const TAG_MASK: u64 = u64::MAX;
        if size <= 1 {
            return value;
        }
        // All-to-all exchange
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&bits.to_le_bytes(), MIN_TAG, &tag_param)
                    .expect("min send");
            }
        }
        let mut recv_buf = [0u8; 8];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker
                    .tag_recv(&mut recv_buf, MIN_TAG, TAG_MASK, &tag_param)
                    .expect("min recv")
                    .expect("min recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                gathered[peer] = u64::from_le_bytes(recv_buf);
            }
        }
        let min_bits = gathered.into_iter().map(|b| f64::from_bits(b)).fold(f64::INFINITY, f64::min);
        drop(summed);
        min_bits
    }

    /// Reduce a f64 value to find the maximum across all ranks (UCX fallback).
    pub fn allreduce_max_f64(&self, value: f64) -> f64 {
        let bits = value.to_bits();
        let rank = self.rank;
        let size = self.size;
        let mut gathered = vec![0u64; size];
        gathered[rank] = bits;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const MAX_TAG: u64 = 0xDEAD0002;
        const TAG_MASK: u64 = u64::MAX;
        if size <= 1 {
            return value;
        }
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&bits.to_le_bytes(), MAX_TAG, &tag_param)
                    .expect("max send");
            }
        }
        let mut recv_buf = [0u8; 8];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker
                    .tag_recv(&mut recv_buf, MAX_TAG, TAG_MASK, &tag_param)
                    .expect("max recv")
                    .expect("max recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                gathered[peer] = u64::from_le_bytes(recv_buf);
            }
        }
        gathered.into_iter().map(|b| f64::from_bits(b)).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Sum a f64 value across all ranks (UCX fallback).
    pub fn allreduce_sum_f64(&self, value: f64) -> f64 {
        let bits = value.to_bits();
        let rank = self.rank;
        let size = self.size;
        let mut gathered = vec![0u64; size];
        gathered[rank] = bits;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const SUM_TAG: u64 = 0xDEAD0003;
        const TAG_MASK: u64 = u64::MAX;
        if size <= 1 {
            return value;
        }
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&bits.to_le_bytes(), SUM_TAG, &tag_param)
                    .expect("sum send");
            }
        }
        let mut recv_buf = [0u8; 8];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker
                    .tag_recv(&mut recv_buf, SUM_TAG, TAG_MASK, &tag_param)
                    .expect("sum recv")
                    .expect("sum recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                gathered[peer] = u64::from_le_bytes(recv_buf);
            }
        }
        gathered.into_iter().map(|b| f64::from_bits(b)).sum()
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

// ============================================================================
// UCC initialization
// ============================================================================

/// Initialize UCC library, context, and team.
///
/// TEMPORARILY DISABLED: UCC team creation segfaults on subsequent collective
/// calls (OOB callback crashes after first call). Return None so all collectives
/// use the UCX fallback methods (ctx.barrier(), ctx.allreduce_*_f64(), etc.).
///
/// The original UCC initialization code is commented out below for reference
/// and future re-enablement once the segfault is resolved.
#[allow(unused_variables)]
fn init_ucc(
    _worker: &worker::Worker,
    _endpoints: &[ep::Ep],
    rank: usize,
    size: usize,
) -> Option<ucc::team::UccTeam> {
    eprintln!(
        "[osu] UCC team creation disabled (segfault workaround) — rank={}, size={}",
        rank, size
    );
    None
}

// ── Thread-local state for OOB callbacks ──
//
// UCC OOB callbacks are invoked synchronously during ucc_context_create /
// ucc_team_create_post. We stash the worker and endpoints in thread-local
// storage so the callbacks can access them without unsafe static lifetimes.
use std::cell::RefCell;

thread_local! {
    static OOB_STATE: RefCell<OobState> = const { RefCell::new(OobState::new()) };
}

struct OobState {
    worker: *const worker::Worker,
    /// Thin pointer to the first element of the endpoints Vec + length.
    /// We store the pointer and length separately to avoid fat pointer issues.
    endpoints_ptr: *const ep::Ep,
    endpoints_len: usize,
    rank: usize,
}

impl OobState {
    const fn new() -> Self {
        Self {
            worker: std::ptr::null(),
            endpoints_ptr: std::ptr::null(),
            endpoints_len: 0,
            rank: 0,
        }
    }
}

// ── UCC OOB Callbacks ──
//
// These callbacks implement the allgather primitive that UCC needs for
// context/team creation. We use UCX tag matching: each rank sends its
// src_buf to all peers and receives from all peers.

/// UCC OOB allgather callback.
///
/// Each rank sends `src_buf` (size bytes) to all other ranks via UCX tag_send,
/// then receives from all other ranks and places the data at offset
/// (peer * size) in `recv_buf`. Our own data is placed at offset (rank * size).
///
/// The `request` output pointer receives a handle to our recv state for later
/// req_test/req_free calls. Since we complete synchronously, request is set to null.
///
/// CRITICAL: This function is called from C code (UCC). We must NEVER panic
/// or unwind across the FFI boundary — all errors are handled gracefully.
///
/// PROTOCOL: To avoid deadlock, we do a phased approach:
///   Phase 1: Post all recv buffers first
///   Phase 2: Send to all peers
///   Phase 3: Wait for all recvs to complete
unsafe extern "C" fn ucc_oob_allgather(
    src_buf: *mut c_void,
    recv_buf: *mut c_void,
    size: usize,
    _allgather_info: *mut c_void,
    request: *mut *mut c_void,
) -> ucc::ucc_status_t {
    // Guard: if size is 0, just return OK (UCC may probe with 0)
    if size == 0 || src_buf.is_null() || recv_buf.is_null() || request.is_null() {
        unsafe {
            if !request.is_null() {
                *request = std::ptr::null_mut();
            }
        }
        return ucc::ucc_status_t_UCC_OK;
    }

    // Debug: write to stderr via libc write (safe in FFI callback)
     let debug_msg = format!(
         "[OOB] rank={} size={} src={:?} recv={:?}\n",
         OOB_STATE.with(|s| s.borrow().rank),
         size,
         src_buf,
         recv_buf
     );
     let _ = unsafe {
         libc::write(
             libc::STDERR_FILENO,
             debug_msg.as_ptr() as *const _,
             debug_msg.len(),
         )
     };

    let result = OOB_STATE.with(|state| {
        let s = state.borrow();
        let worker_ptr = s.worker;
        let endpoints_ptr = s.endpoints_ptr;
        let n_eps = s.endpoints_len;
        let my_rank = s.rank;

        if worker_ptr.is_null() || endpoints_ptr.is_null() || n_eps == 0 {
            return Err(-1isize);
        }

        unsafe {
            let worker = &*worker_ptr;
            let endpoints = std::slice::from_raw_parts(endpoints_ptr, n_eps);

            // Debug: verify endpoints are valid
            let dbg = format!(
                "[OOB] rank={} n_eps={} ep0={:?} ep1={:?}\n",
                my_rank,
                n_eps,
                endpoints[0].handle(),
                if n_eps > 1 {
                    endpoints[1].handle()
                } else {
                    std::ptr::null_mut()
                }
            );            let _ = libc::write(libc::STDERR_FILENO, dbg.as_ptr() as *const _, dbg.len());

            // Place our own data at offset (my_rank * size)
            let my_offset = my_rank * size;
            std::ptr::copy_nonoverlapping(
                src_buf as *const u8,
                (recv_buf as *mut u8).add(my_offset),
                size,
            );

            // Build tag param once
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

            let dbg2 = format!("[OOB] rank={} about to post recvs\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg2.as_ptr() as *const _, dbg2.len());

            // ── Phase 1: Post recv buffers for all peers ──
            // Each peer sends with tag = UCC_OOB_TAG_BASE + peer_rank
            // We need to receive into temp buffers, then copy to recv_buf later
            let mut recv_buffers: Vec<Vec<u8>> = Vec::new();
            let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::new();

            for peer in 0..n_eps {
                if peer == my_rank {
                    recv_buffers.push(Vec::new());
                    recv_reqs.push(None);
                    continue;
                }
                let mut temp_buf = vec![0u8; size];
                let peer_tag = UCC_OOB_TAG_BASE + peer as u64;

                let dbg3 = format!("[OOB] rank={} posting recv from peer={} tag=0x{:x}\n", my_rank, peer, peer_tag);
                let _ = libc::write(libc::STDERR_FILENO, dbg3.as_ptr() as *const _, dbg3.len());

                let recv_result =
                    worker.tag_recv(&mut temp_buf, peer_tag, UCC_OOB_TAG_MASK, &tag_param);
                match recv_result {
                    Ok(Some(req)) => {
                        recv_buffers.push(temp_buf);
                        recv_reqs.push(Some(req));
                    }
                    _ => {
                        // Recv post failed — shouldn't happen
                        return Err(-2isize);
                    }
                }
            }

            let dbg4 = format!("[OOB] rank={} recvs posted, sending\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg4.as_ptr() as *const _, dbg4.len());

            // ── Phase 2: Send our data to all peers ──
            let src_slice = std::slice::from_raw_parts(src_buf as *const u8, size);
            for peer in 0..n_eps {
                if peer == my_rank {
                    continue;
                }
                let tag = UCC_OOB_TAG_BASE + my_rank as u64;
                // Use tag_send (non-sync) with params — tag_send_sync crashes with null param
                match endpoints[peer].tag_send(src_slice, tag, &tag_param) {
                    Ok(Some(send_req)) => {
                        // Progress until send completes
                        while !send_req.check_finished().unwrap_or(false) {
                            let _ = worker.progress();
                        }
                    }
                    Ok(None) => {
                        // Completed immediately (eager path)
                    }
                    Err(_) => {
                        return Err(-3isize);
                    }
                }
            }

            let dbg5 = format!("[OOB] rank={} sends done, waiting recvs\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg5.as_ptr() as *const _, dbg5.len());

            // ── Phase 3: Wait for all recv requests to complete ──
            for peer in 0..n_eps {
                if peer == my_rank {
                    continue;
                }
                if let Some(mut req) = recv_reqs[peer].take() {
                    while !req.check_finished().unwrap_or(false) {
                        let _ = worker.progress();
                    }

                    // Copy received data to the correct offset in recv_buf
                    let peer_offset = peer * size;
                    let dst_ptr = (recv_buf as *mut u8).add(peer_offset);
                    std::ptr::copy_nonoverlapping(recv_buffers[peer].as_ptr(), dst_ptr, size);
                }
            }

            // Set request to null since we completed synchronously
            *request = std::ptr::null_mut();
        }

        Ok(0isize)
    });

    if result == Ok(0) {
        ucc::ucc_status_t_UCC_OK
    } else {
        ucc::ucc_status_t_UCC_ERR_NO_MESSAGE
    }
}
///
/// Since our allgather completes synchronously (request is null),
/// this always returns UCC_OK.
unsafe extern "C" fn ucc_oob_req_test(request: *mut c_void) -> ucc::ucc_status_t {
    if request.is_null() {
        ucc::ucc_status_t_UCC_OK
    } else {
        // If we ever have async requests, test them here
        ucc::ucc_status_t_UCC_OK
    }
}

/// UCC OOB request free callback.
///
/// Since our allgather completes synchronously (request is null),
/// this is a no-op.
unsafe extern "C" fn ucc_oob_req_free(_request: *mut c_void) -> ucc::ucc_status_t {
    // Nothing to free for synchronous completion
    ucc::ucc_status_t_UCC_OK
}
