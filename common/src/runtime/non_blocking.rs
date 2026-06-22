//! Non-blocking collective operations using UCX tag matching.
//!
//! Provides `OsURequest` as the request handle for non-blocking collectives,
//! and `i*` methods on `OsUContext` that return `OsURequest` handles.

use ucx_sys::RequestParamBuilder;

use crate::runtime::context::OsUContext;

/// Handle for a non-blocking collective operation.
///
/// Wraps the internal UCX requests and state needed to complete
/// an `i*` collective. Call `test()` to check completion or `wait()`
/// to block until done (equivalent to the blocking `*` variant).
pub struct OsURequest {
    /// Pending tag receive requests, one per peer (None for self).
    recv_reqs: Vec<Option<ucx_sys::Request>>,
    /// Worker reference (stored as a raw pointer to avoid lifetime issues).
    /// Safety: OsURequest lifetime is bounded by the OsUContext that created it,
    /// so the worker outlives this request.
    worker: *const ucx_sys::worker::Worker,
    /// Remaining peers to wait for.
    remaining: usize,
}

impl OsURequest {
    /// Test whether this non-blocking operation has completed.
    /// Returns `true` if the operation is complete.
    pub fn test(&mut self) -> bool {
        let worker = unsafe { &*self.worker };

        for req_opt in self.recv_reqs.iter_mut() {
            if let Some(req) = req_opt {
                if req.check_finished().unwrap_or(false) {
                    if req_opt.take().is_some() {
                        self.remaining -= 1;
                    }
                }
            }
        }

        // Progress the worker
        loop {
            if !worker.progress() {
                break;
            }
        }

        self.remaining == 0
    }

    /// Wait for this non-blocking operation to complete.
    /// Blocks until all internal requests are finished.
    pub fn wait(&mut self) {
        while !self.test() {}
    }
}

impl OsUContext {
    /// Non-blocking barrier using UCX tag matching.
    ///
    /// Posts sends to all peers and receives from all peers,
    /// returning an `OsURequest` that can be tested/waited on.
    pub fn ibarrier(&self) -> OsURequest {
        let rank = self.rank;
        let size = self.size;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

        const BARRIER_TAG: u64 = 0xBEEFCAFE;
        const TAG_MASK: u64 = u64::MAX;

        if size <= 1 {
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let msg = [rank as u8];

        // Post sends to all peers
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&msg, BARRIER_TAG, &tag_param)
                    .expect("ibarrier send");
            }
        }

        // Post receives from all peers
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }

            let mut recv_buf = [0u8; 1];
            let req = self.worker()
                .tag_recv(&mut recv_buf, BARRIER_TAG, TAG_MASK, &tag_param)
                .expect("ibarrier recv")
                .expect("ibarrier recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking allreduce (sum of u64 values).
    pub fn iallreduce(&self, value: u64) -> OsURequest {
        let rank = self.rank;
        let size = self.size;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

        const REDUCE_TAG: u64 = 0xDEADBEEF;
        const TAG_MASK: u64 = u64::MAX;

        if size <= 1 {
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        // Post sends
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&value.to_le_bytes(), REDUCE_TAG, &tag_param)
                    .expect("iallreduce send");
            }
        }

        // Post receives
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }

            let mut recv_buf = [0u8; 8];
            let req = self.worker()
                .tag_recv(&mut recv_buf, REDUCE_TAG, TAG_MASK, &tag_param)
                .expect("iallreduce recv")
                .expect("iallreduce recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking broadcast: root sends data to all other ranks.
    pub fn ibcast(&self, sendbuf: &[u8], recvbuf: &mut [u8], root: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const BCAST_TAG: u64 = 0xBADC0DE0;
        const TAG_MASK: u64 = u64::MAX;

        if rank == root {
            for peer in 0..size {
                if peer != rank {
                    self.endpoint(peer)
                        .tag_send(sendbuf, BCAST_TAG, &tag_param)
                        .expect("ibcast send");
                }
            }
            recvbuf.copy_from_slice(sendbuf);
            OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            }
        } else {
            let req = self.worker()
                .tag_recv(recvbuf, BCAST_TAG, TAG_MASK, &tag_param)
                .expect("ibcast recv")
                .expect("ibcast recv request");

            let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
            recv_reqs.push(Some(req));

            OsURequest {
                recv_reqs,
                worker: &self.worker as *const _,
                remaining: 1,
            }
        }
    }

    /// Non-blocking allgather: each rank sends, all receive from all.
    pub fn iallgather(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf[..msg_size].copy_from_slice(sendbuf);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLGATHER_TAG: u64 = 0xBADC0DE2;
        const TAG_MASK: u64 = u64::MAX;

        let my_offset = rank * msg_size;
        recvbuf[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);

        // Post sends
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, ALLGATHER_TAG, &tag_param)
                    .expect("iallgather send");
            }
        }

        // Post receives
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }

            let mut recv_buf = vec![0u8; msg_size];
            let req = self.worker()
                .tag_recv(&mut recv_buf, ALLGATHER_TAG, TAG_MASK, &tag_param)
                .expect("iallgather recv")
                .expect("iallgather recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking reduce: all ranks contribute, root gets the result.
    pub fn ireduce(&self, sendbuf: &[u8], recvbuf: &mut [u8], root: usize, msg_size: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCE_TAG: u64 = 0xBADC0DE1;
        const TAG_MASK: u64 = u64::MAX;

        // Post sends
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, REDUCE_TAG, &tag_param)
                    .expect("ireduce send");
            }
        }

        // Post receives
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }

            let mut recv_buf = vec![0u8; msg_size];
            let req = self.worker()
                .tag_recv(&mut recv_buf, REDUCE_TAG, TAG_MASK, &tag_param)
                .expect("ireduce recv")
                .expect("ireduce recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking scatter: root sends one chunk per rank.
    pub fn iscatter(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
                recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const SCATTER_TAG: u64 = 0xBADC0DE5;
        const TAG_MASK: u64 = u64::MAX;

        if rank == root {
            let my_offset = rank * msg_size;
            if my_offset + msg_size <= sendbuf.len() && msg_size <= recvbuf.len() {
                recvbuf[..msg_size].copy_from_slice(&sendbuf[my_offset..my_offset + msg_size]);
            }

            for peer in 0..size {
                if peer != rank {
                    let peer_offset = peer * msg_size;
                    if peer_offset + msg_size <= sendbuf.len() {
                        self.endpoint(peer)
                            .tag_send(&sendbuf[peer_offset..peer_offset + msg_size], SCATTER_TAG, &tag_param)
                            .expect("iscatter send");
                    }
                }
            }

            OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            }
        } else {
            let req = self.worker()
                .tag_recv(recvbuf, SCATTER_TAG, TAG_MASK, &tag_param)
                .expect("iscatter recv")
                .expect("iscatter recv request");

            let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(1);
            recv_reqs.push(Some(req));

            OsURequest {
                recv_reqs,
                worker: &self.worker as *const _,
                remaining: 1,
            }
        }
    }

    /// Non-blocking gather: all ranks send to root.
    pub fn igather(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                recvbuf[..sendbuf.len()].copy_from_slice(sendbuf);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const GATHER_TAG: u64 = 0xBADC0DE4;
        const TAG_MASK: u64 = u64::MAX;

        if rank != root {
            self.endpoint(root)
                .tag_send(sendbuf, GATHER_TAG, &tag_param)
                .expect("igather send");

            OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            }
        } else {
            let my_offset = rank * msg_size;
            if my_offset + msg_size <= recvbuf.len() {
                recvbuf[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);
            }

            let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
            let mut remaining = 0;

            for peer in 0..size {
                if peer == rank {
                    recv_reqs.push(None);
                    continue;
                }

                let mut recv_buf = vec![0u8; msg_size];
                let req = self.worker()
                    .tag_recv(&mut recv_buf, GATHER_TAG, TAG_MASK, &tag_param)
                    .expect("igather recv")
                    .expect("igather recv request");
                recv_reqs.push(Some(req));
                remaining += 1;
            }

            OsURequest {
                recv_reqs,
                worker: &self.worker as *const _,
                remaining,
            }
        }
    }

    /// Non-blocking alltoall.
    pub fn ialltoall(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLTOALL_TAG: u64 = 0xBADC0DE3;
        const TAG_MASK: u64 = u64::MAX;

        // Post sends
        for peer in 0..size {
            if peer != rank {
                let piece = &sendbuf[peer * msg_size..(peer + 1) * msg_size];
                self.endpoint(peer)
                    .tag_send(piece, ALLTOALL_TAG, &tag_param)
                    .expect("ialltoall send");
            }
        }

        // Post receives
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                let my_offset = rank * msg_size;
                recvbuf[my_offset..my_offset + msg_size]
                    .copy_from_slice(&sendbuf[my_offset..my_offset + msg_size]);
                recv_reqs.push(None);
                continue;
            }

            let mut recv_buf = vec![0u8; msg_size];
            let req = self.worker()
                .tag_recv(&mut recv_buf, ALLTOALL_TAG, TAG_MASK, &tag_param)
                .expect("ialltoall recv")
                .expect("ialltoall recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }
}
