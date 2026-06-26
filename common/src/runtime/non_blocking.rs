//! Non-blocking collective operations using UCX tag matching.
//!
//! `needless_range_loop` is allowed because `for peer in 0..size` is the
//! idiomatic MPI-style rank iteration pattern used throughout this module.
//!
//! Provides `OsURequest` as the request handle for non-blocking collectives,
//! and `i*` methods on `OsUContext` that return `OsURequest` handles.

#![allow(clippy::needless_range_loop)]

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
            if let Some(req) = req_opt
                && req.check_finished().unwrap_or(false)
                && req_opt.take().is_some()
            {
                self.remaining -= 1;
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
            let req = self
                .worker()
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
            let req = self
                .worker()
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
            let req = self
                .worker()
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
            let req = self
                .worker()
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
    pub fn ireduce(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        _root: usize,
        msg_size: usize,
    ) -> OsURequest {
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
            let req = self
                .worker()
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
    pub fn iscatter(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
        root: usize,
    ) -> OsURequest {
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
                            .tag_send(
                                &sendbuf[peer_offset..peer_offset + msg_size],
                                SCATTER_TAG,
                                &tag_param,
                            )
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
            let req = self
                .worker()
                .tag_recv(recvbuf, SCATTER_TAG, TAG_MASK, &tag_param)
                .expect("iscatter recv")
                .expect("iscatter recv request");

            let recv_reqs: Vec<Option<ucx_sys::Request>> = vec![Some(req)];

            OsURequest {
                recv_reqs,
                worker: &self.worker as *const _,
                remaining: 1,
            }
        }
    }

    /// Non-blocking gather: all ranks send to root.
    pub fn igather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
        root: usize,
    ) -> OsURequest {
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
                let req = self
                    .worker()
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
            let req = self
                .worker()
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

    /// Non-blocking allgatherv: each rank sends, all receive with variable counts/displacements.
    pub fn iallgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let offset = displs[rank];
                let len = counts[rank];
                let copy_len = len
                    .min(sendbuf.len())
                    .min(recvbuf.len().saturating_sub(offset));
                recvbuf[offset..offset + copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLGATHERV_TAG: u64 = 0xBADC0DE8;
        const TAG_MASK: u64 = u64::MAX;

        let my_offset = displs[rank];
        let my_len = counts[rank];
        if my_offset + my_len <= recvbuf.len() {
            recvbuf[my_offset..my_offset + my_len].copy_from_slice(sendbuf);
        }

        // Post sends to all peers
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, ALLGATHERV_TAG, &tag_param)
                    .expect("iallgatherv send");
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

            let len = counts[peer];
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, ALLGATHERV_TAG, TAG_MASK, &tag_param)
                .expect("iallgatherv recv")
                .expect("iallgatherv recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking gatherv: all ranks send to root with variable counts/displacements.
    pub fn igatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: usize,
        recv_count: usize,
        send_count: usize,
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let offset = rank * recv_count;
                let len = send_count
                    .min(recv_count)
                    .min(sendbuf.len())
                    .min(recvbuf.len().saturating_sub(offset));
                if len > 0 {
                    recvbuf[offset..offset + len].copy_from_slice(&sendbuf[..len]);
                }
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const GATHERV_TAG: u64 = 0xBADC0DE6;
        const TAG_MASK: u64 = u64::MAX;

        if rank != root {
            self.endpoint(root)
                .tag_send(sendbuf, GATHERV_TAG, &tag_param)
                .expect("igatherv send");

            OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            }
        } else {
            let my_displ = rank * recv_count;
            if my_displ + send_count <= recvbuf.len() {
                recvbuf[my_displ..my_displ + send_count].copy_from_slice(sendbuf);
            }

            let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
            let mut remaining = 0;

            for peer in 0..size {
                if peer == rank {
                    recv_reqs.push(None);
                    continue;
                }

                let mut recv_buf = vec![0u8; recv_count];
                let req = self
                    .worker()
                    .tag_recv(&mut recv_buf, GATHERV_TAG, TAG_MASK, &tag_param)
                    .expect("igatherv recv")
                    .expect("igatherv recv request");
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

    /// Non-blocking scatterv: root sends variable-count data to all ranks.
    pub fn iscatterv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
        root: usize,
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let offset = displs[rank];
                let len = counts[rank];
                let copy_len = len
                    .min(sendbuf.len().saturating_sub(offset))
                    .min(recvbuf.len());
                recvbuf[..copy_len].copy_from_slice(&sendbuf[offset..offset + copy_len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const SCATTERV_TAG: u64 = 0xBADC0DE7;
        const TAG_MASK: u64 = u64::MAX;

        if rank == root {
            let my_offset = displs[rank];
            let my_len = counts[rank];
            if my_offset + my_len <= sendbuf.len() && my_len <= recvbuf.len() {
                recvbuf[..my_len].copy_from_slice(&sendbuf[my_offset..my_offset + my_len]);
            }

            for peer in 0..size {
                if peer != rank {
                    let offset = displs[peer];
                    let len = counts[peer];
                    if offset + len <= sendbuf.len() {
                        self.endpoint(peer)
                            .tag_send(&sendbuf[offset..offset + len], SCATTERV_TAG, &tag_param)
                            .expect("iscatterv send");
                    }
                }
            }

            OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            }
        } else {
            let len = counts[rank];
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, SCATTERV_TAG, TAG_MASK, &tag_param)
                .expect("iscatterv recv")
                .expect("iscatterv recv request");

            let recv_reqs: Vec<Option<ucx_sys::Request>> = vec![Some(req)];

            OsURequest {
                recv_reqs,
                worker: &self.worker as *const _,
                remaining: 1,
            }
        }
    }

    /// Non-blocking alltoallv: each rank sends variable-size pieces to every peer.
    pub fn ialltoallv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        recv_counts: &[usize],
        send_displs: &[usize],
        recv_displs: &[usize],
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            for p in 0..size {
                let src_off = send_displs[p];
                let dst_off = recv_displs[p];
                let len = send_counts[p];
                if src_off + len <= sendbuf.len() && dst_off + len <= recvbuf.len() {
                    recvbuf[dst_off..dst_off + len]
                        .copy_from_slice(&sendbuf[src_off..src_off + len]);
                }
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLTOALLV_TAG: u64 = 0xBADC0DEA;
        const TAG_MASK: u64 = u64::MAX;

        // Post sends to all peers
        for peer in 0..size {
            if peer != rank {
                let src_off = send_displs[peer];
                let len = send_counts[peer];
                if src_off + len <= sendbuf.len() {
                    self.endpoint(peer)
                        .tag_send(&sendbuf[src_off..src_off + len], ALLTOALLV_TAG, &tag_param)
                        .expect("ialltoallv send");
                }
            }
        }

        // Post receives from all peers
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut remaining = 0;

        for peer in 0..size {
            if peer == rank {
                let src_off = send_displs[peer];
                let dst_off = recv_displs[peer];
                let len = send_counts[peer];
                if src_off + len <= sendbuf.len() && dst_off + len <= recvbuf.len() {
                    recvbuf[dst_off..dst_off + len]
                        .copy_from_slice(&sendbuf[src_off..src_off + len]);
                }
                recv_reqs.push(None);
                continue;
            }

            let len = recv_counts[peer];
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, ALLTOALLV_TAG, TAG_MASK, &tag_param)
                .expect("ialltoallv recv")
                .expect("ialltoallv recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking alltoallw: like alltoallv but with per-peer datatypes.
    /// Since we only use bytes, this is identical to ialltoallv.
    pub fn ialltoallw(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        recv_counts: &[usize],
        send_displs: &[usize],
        recv_displs: &[usize],
    ) -> OsURequest {
        self.ialltoallv(
            sendbuf,
            recvbuf,
            send_counts,
            recv_counts,
            send_displs,
            recv_displs,
        )
    }

    /// Non-blocking reduce_scatter: all ranks send, each rank receives a portion.
    pub fn ireduce_scatter(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        recvcounts: &[usize],
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let len = recvcounts[rank].min(sendbuf.len()).min(recvbuf.len());
                recvbuf[..len].copy_from_slice(&sendbuf[..len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        // Build displs from recvcounts
        let mut displs: Vec<usize> = Vec::with_capacity(size);
        let mut d = 0;
        for &c in recvcounts {
            displs.push(d);
            d += c;
        }
        let total = d;

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCESCATTER_TAG: u64 = 0xBADC0DE9;
        const TAG_MASK: u64 = u64::MAX;

        // Post sends to all peers
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, REDUCESCATTER_TAG, &tag_param)
                    .expect("ireduce_scatter send");
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

            let mut recv_buf = vec![0u8; total];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, REDUCESCATTER_TAG, TAG_MASK, &tag_param)
                .expect("ireduce_scatter recv")
                .expect("ireduce_scatter recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    /// Non-blocking reduce_scatter_block: all ranks send `elemcount` bytes; each rank receives `elemcount / numprocs`.
    pub fn ireduce_scatter_block(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        elemcount: usize,
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let len = elemcount.min(sendbuf.len()).min(recvbuf.len());
            if len > 0 {
                recvbuf[..len].copy_from_slice(&sendbuf[..len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCE_SCATTER_BLOCK_TAG: u64 = 0xBADC0DEB;
        const TAG_MASK: u64 = u64::MAX;

        // Post sends to all peers
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, REDUCE_SCATTER_BLOCK_TAG, &tag_param)
                    .expect("ireduce_scatter_block send");
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

            let mut recv_buf = vec![0u8; elemcount];
            let req = self
                .worker()
                .tag_recv(
                    &mut recv_buf,
                    REDUCE_SCATTER_BLOCK_TAG,
                    TAG_MASK,
                    &tag_param,
                )
                .expect("ireduce_scatter_block recv")
                .expect("ireduce_scatter_block recv request");
            recv_reqs.push(Some(req));
            remaining += 1;
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining,
        }
    }

    // -----------------------------------------------------------------------
    // Neighbor collectives (ring topology: each rank connects to prev/next)
    // -----------------------------------------------------------------------

    /// Compute ring neighbors for this rank. Returns (sources, destinations).
    fn neighbor_ring(&self) -> (Vec<usize>, Vec<usize>) {
        let rank = self.rank;
        let size = self.size;
        if size <= 2 {
            return (vec![1 ^ rank], vec![1 ^ rank]);
        }
        let prev = (rank.wrapping_sub(1)).rem_euclid(size);
        let next = (rank + 1).rem_euclid(size);
        (vec![prev], vec![next])
    }

    /// Non-blocking neighbor allgather (ring topology).
    /// Each rank sends to and receives from its neighbors only.
    pub fn ineighbor_allgather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
            recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const NHBR_ALLGATHER_TAG: u64 = 0x01010001;
        const TAG_MASK: u64 = u64::MAX;

        let (sources, destinations) = self.neighbor_ring();
        let num_neighbors = sources.len();

        // Copy own contribution into recvbuf
        let my_offset = rank * msg_size;
        if my_offset + msg_size <= recvbuf.len() {
            recvbuf[my_offset..my_offset + msg_size]
                .copy_from_slice(&sendbuf[..msg_size.min(sendbuf.len())]);
        }

        // Post sends to destinations
        for &dst in &destinations {
            self.endpoint(dst)
                .tag_send(
                    &sendbuf[..msg_size.min(sendbuf.len())],
                    NHBR_ALLGATHER_TAG,
                    &tag_param,
                )
                .expect("ineighbor_allgather send");
        }

        // Post receives from sources
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(num_neighbors);
        for &_src in &sources {
            let mut recv_buf = vec![0u8; msg_size];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, NHBR_ALLGATHER_TAG, TAG_MASK, &tag_param)
                .expect("ineighbor_allgather recv")
                .expect("ineighbor_allgather recv request");
            recv_reqs.push(Some(req));
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining: num_neighbors,
        }
    }

    /// Non-blocking neighbor allgatherv (ring topology).
    pub fn ineighbor_allgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
    ) -> OsURequest {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let offset = displs.get(rank).copied().unwrap_or(0);
            let len = counts.get(rank).copied().unwrap_or(0);
            let copy_len = len
                .min(sendbuf.len())
                .min(recvbuf.len().saturating_sub(offset));
            if copy_len > 0 {
                recvbuf[offset..offset + copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const NHBR_ALLGATHERV_TAG: u64 = 0x01010002;
        const TAG_MASK: u64 = u64::MAX;

        let (sources, destinations) = self.neighbor_ring();
        let num_neighbors = sources.len();

        // Copy own contribution
        let my_offset = displs.get(rank).copied().unwrap_or(0);
        let my_len = counts.get(rank).copied().unwrap_or(0);
        if my_offset + my_len <= recvbuf.len() {
            recvbuf[my_offset..my_offset + my_len]
                .copy_from_slice(&sendbuf[..my_len.min(sendbuf.len())]);
        }

        // Post sends to destinations
        for &dst in &destinations {
            let len = counts.get(rank).copied().unwrap_or(0);
            self.endpoint(dst)
                .tag_send(
                    &sendbuf[..len.min(sendbuf.len())],
                    NHBR_ALLGATHERV_TAG,
                    &tag_param,
                )
                .expect("ineighbor_allgatherv send");
        }

        // Post receives from sources
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(num_neighbors);
        for &src in &sources {
            let len = counts.get(src).copied().unwrap_or(0);
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, NHBR_ALLGATHERV_TAG, TAG_MASK, &tag_param)
                .expect("ineighbor_allgatherv recv")
                .expect("ineighbor_allgatherv recv request");
            recv_reqs.push(Some(req));
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining: num_neighbors,
        }
    }

    /// Non-blocking neighbor alltoall (ring topology).
    pub fn ineighbor_alltoall(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
    ) -> OsURequest {
        let _rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
            recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const NHBR_ALLTOALL_TAG: u64 = 0x01010003;
        const TAG_MASK: u64 = u64::MAX;

        let (sources, destinations) = self.neighbor_ring();
        let num_neighbors = sources.len();

        // Post sends to destinations
        for &dst in &destinations {
            let offset = dst * msg_size;
            let end = (offset + msg_size).min(sendbuf.len());
            if offset < sendbuf.len() {
                self.endpoint(dst)
                    .tag_send(&sendbuf[offset..end], NHBR_ALLTOALL_TAG, &tag_param)
                    .expect("ineighbor_alltoall send");
            }
        }

        // Post receives from sources
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(num_neighbors);
        for &_src in &sources {
            let mut recv_buf = vec![0u8; msg_size];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, NHBR_ALLTOALL_TAG, TAG_MASK, &tag_param)
                .expect("ineighbor_alltoall recv")
                .expect("ineighbor_alltoall recv request");
            recv_reqs.push(Some(req));
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining: num_neighbors,
        }
    }

    /// Non-blocking neighbor alltoallv (ring topology).
    pub fn ineighbor_alltoallv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        recv_counts: &[usize],
        send_displs: &[usize],
        recv_displs: &[usize],
    ) -> OsURequest {
        let _rank = self.rank;
        let size = self.size;

        if size <= 1 {
            for p in 0..size {
                let src_off = send_displs.get(p).copied().unwrap_or(0);
                let dst_off = recv_displs.get(p).copied().unwrap_or(0);
                let len = send_counts.get(p).copied().unwrap_or(0);
                let copy = len
                    .min(sendbuf.len().saturating_sub(src_off))
                    .min(recvbuf.len().saturating_sub(dst_off));
                if copy > 0 {
                    recvbuf[dst_off..dst_off + copy]
                        .copy_from_slice(&sendbuf[src_off..src_off + copy]);
                }
            }
            return OsURequest {
                recv_reqs: Vec::new(),
                worker: std::ptr::null(),
                remaining: 0,
            };
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const NHBR_ALLTOALLV_TAG: u64 = 0x01010004;
        const TAG_MASK: u64 = u64::MAX;

        let (sources, destinations) = self.neighbor_ring();
        let num_neighbors = sources.len();

        // Post sends to destinations
        for &dst in &destinations {
            let src_off = send_displs.get(dst).copied().unwrap_or(0);
            let len = send_counts.get(dst).copied().unwrap_or(0);
            let end = (src_off + len).min(sendbuf.len());
            if src_off < sendbuf.len() {
                self.endpoint(dst)
                    .tag_send(&sendbuf[src_off..end], NHBR_ALLTOALLV_TAG, &tag_param)
                    .expect("ineighbor_alltoallv send");
            }
        }

        // Post receives from sources
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(num_neighbors);
        for &src in &sources {
            let len = recv_counts.get(src).copied().unwrap_or(0);
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, NHBR_ALLTOALLV_TAG, TAG_MASK, &tag_param)
                .expect("ineighbor_alltoallv recv")
                .expect("ineighbor_alltoallv recv request");
            recv_reqs.push(Some(req));
        }

        OsURequest {
            recv_reqs,
            worker: &self.worker as *const _,
            remaining: num_neighbors,
        }
    }

    /// Non-blocking neighbor alltoallw (ring topology).
    /// Same as alltoallv since we only use bytes.
    pub fn ineighbor_alltoallw(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        recv_counts: &[usize],
        send_displs: &[usize],
        recv_displs: &[usize],
    ) -> OsURequest {
        self.ineighbor_alltoallv(
            sendbuf,
            recvbuf,
            send_counts,
            recv_counts,
            send_displs,
            recv_displs,
        )
    }
}
