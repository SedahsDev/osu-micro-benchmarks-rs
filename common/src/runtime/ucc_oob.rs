//! UCC initialization and OOB (out-of-band) callbacks.
//!
//! UCC is temporarily disabled — team creation segfaults. This module is kept
//! for future re-enablement once the segfault is resolved.

use std::cell::RefCell;
use std::os::raw::c_void;

use ucx_sys::ep;
use ucx_sys::worker;
use ucx_sys::RequestParamBuilder;

use crate::runtime::constants::*;

/// Initialize UCC library, context, and team.
///
/// TEMPORARILY DISABLED: UCC team creation segfaults on subsequent collective
/// calls (OOB callback crashes after first call). Return None so all collectives
/// use the UCX fallback methods.
#[allow(unused_variables)]
pub fn init_ucc(
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

thread_local! {
    static OOB_STATE: RefCell<OobState> = const { RefCell::new(OobState::new()) };
}

struct OobState {
    worker: *const worker::Worker,
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
// context/team creation. We use UCX tag matching.

/// UCC OOB allgather callback.
#[allow(unused_variables)]
unsafe extern "C" fn ucc_oob_allgather(
    src_buf: *mut c_void,
    recv_buf: *mut c_void,
    size: usize,
    _allgather_info: *mut c_void,
    request: *mut *mut c_void,
) -> ucc::ucc_status_t {
    if size == 0 || src_buf.is_null() || recv_buf.is_null() || request.is_null() {
        unsafe {
            if !request.is_null() {
                *request = std::ptr::null_mut();
            }
        }
        return ucc::ucc_status_t_UCC_OK;
    }

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
            );
            let _ = libc::write(libc::STDERR_FILENO, dbg.as_ptr() as *const _, dbg.len());

            // Place our own data at offset (my_rank * size)
            let my_offset = my_rank * size;
            std::ptr::copy_nonoverlapping(
                src_buf as *const u8,
                (recv_buf as *mut u8).add(my_offset),
                size,
            );

            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

            let dbg2 = format!("[OOB] rank={} about to post recvs\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg2.as_ptr() as *const _, dbg2.len());

            // Phase 1: Post recv buffers for all peers
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
                        return Err(-2isize);
                    }
                }
            }

            let dbg4 = format!("[OOB] rank={} recvs posted, sending\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg4.as_ptr() as *const _, dbg4.len());

            // Phase 2: Send our data to all peers
            let src_slice = std::slice::from_raw_parts(src_buf as *const u8, size);
            for peer in 0..n_eps {
                if peer == my_rank {
                    continue;
                }
                let tag = UCC_OOB_TAG_BASE + my_rank as u64;
                match endpoints[peer].tag_send(src_slice, tag, &tag_param) {
                    Ok(Some(send_req)) => {
                        while !send_req.check_finished().unwrap_or(false) {
                            let _ = worker.progress();
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        return Err(-3isize);
                    }
                }
            }

            let dbg5 = format!("[OOB] rank={} sends done, waiting recvs\n", my_rank);
            let _ = libc::write(libc::STDERR_FILENO, dbg5.as_ptr() as *const _, dbg5.len());

            // Phase 3: Wait for all recv requests to complete
            for peer in 0..n_eps {
                if peer == my_rank {
                    continue;
                }
                if let Some(mut req) = recv_reqs[peer].take() {
                    while !req.check_finished().unwrap_or(false) {
                        let _ = worker.progress();
                    }
                    let peer_offset = peer * size;
                    let dst_ptr = (recv_buf as *mut u8).add(peer_offset);
                    std::ptr::copy_nonoverlapping(recv_buffers[peer].as_ptr(), dst_ptr, size);
                }
            }

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

/// UCC OOB request test callback.
#[allow(unused_variables)]
unsafe extern "C" fn ucc_oob_req_test(request: *mut c_void) -> ucc::ucc_status_t {
    if request.is_null() {
        ucc::ucc_status_t_UCC_OK
    } else {
        ucc::ucc_status_t_UCC_OK
    }
}

/// UCC OOB request free callback.
#[allow(unused_variables)]
unsafe extern "C" fn ucc_oob_req_free(_request: *mut c_void) -> ucc::ucc_status_t {
    ucc::ucc_status_t_UCC_OK
}
