//! Blocking collective operations implemented via UCX tag-matching fallback.
//!
//! `needless_range_loop` is allowed because `for peer in 0..size` is the
//! idiomatic MPI-style rank iteration pattern used throughout this module.

#![allow(clippy::needless_range_loop)]

use ucx_sys::RequestParamBuilder;

use crate::runtime::context::OsUContext;

impl OsUContext {
    /// Try the OpenSHMEM byte-sum allreduce. Returns false when the optional
    /// runtime is unavailable so callers can retain their UCX implementation.
    pub fn openshmem_allreduce(&self, sendbuf: &[u8], recvbuf: &mut [u8]) -> bool {
        if !self.openshmem_initialized || sendbuf.len() != recvbuf.len() {
            return false;
        }
        recvbuf.copy_from_slice(sendbuf);
        openshmem::coll::reduce(ucc::collective::UccReductionOp::Sum, recvbuf).is_ok()
    }

    /// Simple barrier using UCX tag matching (all-to-all handshake).
    pub fn barrier(&self) {
        if self.openshmem_initialized && openshmem::coll::barrier().is_ok() {
            return;
        }
        let rank = self.rank;
        let size = self.size;
        let worker = &self.worker;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();

        const BARRIER_TAG: u64 = 0xBEEFCAFE;
        const TAG_MASK: u64 = u64::MAX;

        if size <= 1 {
            return;
        }

        let msg = [rank as u8];
        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&msg, BARRIER_TAG, &tag_param)
                    .expect("barrier send");
            }
        }

        let mut recv_buf = [0u8; 1];
        for peer in 0..size {
            if peer != rank {
                let req = worker
                    .tag_recv(&mut recv_buf, BARRIER_TAG, TAG_MASK, &tag_param)
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
        let _summed = self.allreduce_u64(bits);
        let rank = self.rank;
        let size = self.size;
        let mut gathered = vec![0u64; size];
        gathered[rank] = bits;
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const MIN_TAG: u64 = 0xDEAD0001;
        const TAG_MASK: u64 = u64::MAX;
        if size <= 1 {
            return value;
        }
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
        gathered
            .into_iter()
            .map(f64::from_bits)
            .fold(f64::INFINITY, f64::min)
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
        gathered
            .into_iter()
            .map(f64::from_bits)
            .fold(f64::NEG_INFINITY, f64::max)
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
        gathered.into_iter().map(f64::from_bits).sum()
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

        let mut gathered = vec![0u64; size];
        gathered[rank] = value;

        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(&value.to_le_bytes(), REDUCE_TAG, &tag_param)
                    .expect("reduce send");
            }
        }

        let mut recv_buf = [0u8; 8];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker
                    .tag_recv(&mut recv_buf, REDUCE_TAG, TAG_MASK, &tag_param)
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

    /// Broadcast: root sends data to all other ranks (UCX tag-matching fallback).
    pub fn bcast(&self, sendbuf: &[u8], recvbuf: &mut [u8], root: usize) {
        if self.openshmem_initialized {
            recvbuf.copy_from_slice(sendbuf);
            if openshmem::coll::broadcast(root, recvbuf).is_ok() {
                return;
            }
        }
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return;
        }

        if rank == root {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const BCAST_TAG: u64 = 0xBADC0DE0;
            for peer in 0..size {
                if peer != rank {
                    self.endpoint(peer)
                        .tag_send(sendbuf, BCAST_TAG, &tag_param)
                        .expect("bcast send");
                }
            }
            recvbuf.copy_from_slice(sendbuf);
        } else {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const BCAST_TAG: u64 = 0xBADC0DE0;
            const TAG_MASK: u64 = u64::MAX;
            let req = self
                .worker()
                .tag_recv(recvbuf, BCAST_TAG, TAG_MASK, &tag_param)
                .expect("bcast recv")
                .expect("bcast recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
        }
    }

    /// Reduce: all ranks contribute, root gets the result (UCX tag-matching fallback).
    pub fn reduce(&self, sendbuf: &[u8], recvbuf: &mut [u8], root: usize) {
        let rank = self.rank;
        let size = self.size;
        let msg_size = sendbuf.len();

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return;
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCE_TAG: u64 = 0xBADC0DE1;
        const TAG_MASK: u64 = u64::MAX;

        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, REDUCE_TAG, &tag_param)
                    .expect("reduce send");
            }
        }

        let mut gathered = vec![0u8; msg_size * size];
        let my_offset = rank * msg_size;
        gathered[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);

        let mut recv_buf = vec![0u8; msg_size];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker()
                    .tag_recv(&mut recv_buf, REDUCE_TAG, TAG_MASK, &tag_param)
                    .expect("reduce recv")
                    .expect("reduce recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                let peer_offset = peer * msg_size;
                gathered[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
            }
        }

        if rank == root {
            for (i, item) in recvbuf.iter_mut().enumerate().take(msg_size) {
                let sum: u16 = (0..size).map(|r| gathered[r * msg_size + i] as u16).sum();
                *item = (sum % 256) as u8;
            }
        }
    }

    /// Allgather: each rank sends its data, all ranks receive from all (UCX tag-matching fallback).
    pub fn allgather(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
        if self.openshmem_initialized {
            if let Ok(values) = openshmem::coll::collect(sendbuf) {
                if values.len() == recvbuf.len() {
                    recvbuf.copy_from_slice(&values);
                    return;
                }
            }
        }
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf[..msg_size].copy_from_slice(sendbuf);
            return;
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLGATHER_TAG: u64 = 0xBADC0DE2;
        const TAG_MASK: u64 = u64::MAX;

        let my_offset = rank * msg_size;
        recvbuf[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);

        for peer in 0..size {
            if peer != rank {
                self.endpoint(peer)
                    .tag_send(sendbuf, ALLGATHER_TAG, &tag_param)
                    .expect("allgather send");
            }
        }

        let mut recv_buf = vec![0u8; msg_size];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker()
                    .tag_recv(&mut recv_buf, ALLGATHER_TAG, TAG_MASK, &tag_param)
                    .expect("allgather recv")
                    .expect("allgather recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                let peer_offset = peer * msg_size;
                recvbuf[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
            }
        }
    }

    /// Alltoall: each rank sends a piece to every peer, receives from every peer.
    /// TODO(openshmem): `openshmem::coll` has no alltoall API; retain UCX fallback.
    pub fn alltoall(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            recvbuf.copy_from_slice(sendbuf);
            return;
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLTOALL_TAG: u64 = 0xBADC0DE3;
        const TAG_MASK: u64 = u64::MAX;

        for peer in 0..size {
            if peer != rank {
                let piece = &sendbuf[peer * msg_size..(peer + 1) * msg_size];
                self.endpoint(peer)
                    .tag_send(piece, ALLTOALL_TAG, &tag_param)
                    .expect("alltoall send");
            }
        }

        let mut recv_buf = vec![0u8; msg_size];
        for peer in 0..size {
            if peer != rank {
                let req = self
                    .worker()
                    .tag_recv(&mut recv_buf, ALLTOALL_TAG, TAG_MASK, &tag_param)
                    .expect("alltoall recv")
                    .expect("alltoall recv request");
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                let peer_offset = peer * msg_size;
                recvbuf[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
            } else {
                let my_offset = rank * msg_size;
                recvbuf[my_offset..my_offset + msg_size]
                    .copy_from_slice(&sendbuf[my_offset..my_offset + msg_size]);
            }
        }
    }

    /// Gather: all ranks send `msg_size` bytes to root.
    // TODO(openshmem): coll currently exposes no root-directed gather.
    /// TODO(openshmem): `openshmem::coll` has no gather API; retain UCX fallback.
    pub fn gather(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                recvbuf[..sendbuf.len()].copy_from_slice(sendbuf);
            }
            return;
        }

        if rank != root {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const GATHER_TAG: u64 = 0xBADC0DE4;
            self.endpoint(root)
                .tag_send(sendbuf, GATHER_TAG, &tag_param)
                .expect("gather send");
        } else {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const GATHER_TAG: u64 = 0xBADC0DE4;
            const TAG_MASK: u64 = u64::MAX;

            let my_offset = rank * msg_size;
            if my_offset + msg_size <= recvbuf.len() {
                recvbuf[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);
            }

            let mut recv_buf = vec![0u8; msg_size];
            for peer in 0..size {
                if peer != rank {
                    let req = self
                        .worker()
                        .tag_recv(&mut recv_buf, GATHER_TAG, TAG_MASK, &tag_param)
                        .expect("gather recv")
                        .expect("gather recv request");
                    while !req.check_finished().unwrap_or(false) {
                        self.progress();
                    }
                    let peer_offset = peer * msg_size;
                    if peer_offset + msg_size <= recvbuf.len() {
                        recvbuf[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
                    }
                }
            }
        }
    }

    /// Scatter: root sends `msg_size * size` bytes (one chunk per rank).
    // TODO(openshmem): coll currently exposes no root-directed scatter.
    /// TODO(openshmem): `openshmem::coll` has no scatter API; retain UCX fallback.
    pub fn scatter(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
                recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return;
        }

        if rank == root {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const SCATTER_TAG: u64 = 0xBADC0DE5;

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
                            .expect("scatter send");
                    }
                }
            }
        } else {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const SCATTER_TAG: u64 = 0xBADC0DE5;
            const TAG_MASK: u64 = u64::MAX;

            let req = self
                .worker()
                .tag_recv(recvbuf, SCATTER_TAG, TAG_MASK, &tag_param)
                .expect("scatter recv")
                .expect("scatter recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
        }
    }

    /// Gatherv: all ranks send `msg_size` bytes to root; root receives into variable-count slots.
    /// TODO(openshmem): variable-count collectives are not exposed by the current coll API.
    pub fn gatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
        root: usize,
    ) {
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
            return;
        }

        if rank != root {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const GATHERV_TAG: u64 = 0xBADC0DE6;
            let req = self
                .endpoint(root)
                .tag_send(sendbuf, GATHERV_TAG, &tag_param);
            if let Ok(Some(req)) = req {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        } else {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const GATHERV_TAG: u64 = 0xBADC0DE6;
            const TAG_MASK: u64 = u64::MAX;

            let my_offset = displs[rank];
            let my_len = counts[rank];
            if my_offset + my_len <= recvbuf.len() {
                recvbuf[my_offset..my_offset + my_len].copy_from_slice(sendbuf);
            }

            for peer in 0..size {
                if peer == rank {
                    continue;
                }
                let offset = displs[peer];
                let len = counts[peer];
                let mut recv_buf = vec![0u8; len];
                let req = self
                    .worker()
                    .tag_recv(&mut recv_buf, GATHERV_TAG, TAG_MASK, &tag_param);
                if let Ok(Some(req)) = req {
                    while !req.check_finished().unwrap_or(false) {
                        self.progress();
                    }
                    if offset + len <= recvbuf.len() {
                        recvbuf[offset..offset + len].copy_from_slice(&recv_buf);
                    }
                }
            }
        }
    }

    /// Scatterv: root sends variable-count data; each rank receives `counts[rank]` bytes.
    /// TODO(openshmem): variable-count collectives are not exposed by the current coll API.
    pub fn scatterv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
        root: usize,
    ) {
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
            return;
        }

        if rank == root {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const SCATTERV_TAG: u64 = 0xBADC0DE7;

            let my_offset = displs[rank];
            let my_len = counts[rank];
            if my_offset + my_len <= sendbuf.len() && my_len <= recvbuf.len() {
                recvbuf[..my_len].copy_from_slice(&sendbuf[my_offset..my_offset + my_len]);
            }

            for peer in 0..size {
                if peer == rank {
                    continue;
                }
                let offset = displs[peer];
                let len = counts[peer];
                if offset + len <= sendbuf.len() {
                    let req = self.endpoint(peer).tag_send(
                        &sendbuf[offset..offset + len],
                        SCATTERV_TAG,
                        &tag_param,
                    );
                    if let Ok(Some(req)) = req {
                        while !req.check_finished().unwrap_or(false) {
                            self.progress();
                        }
                    }
                }
            }
        } else {
            let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
            const SCATTERV_TAG: u64 = 0xBADC0DE7;
            const TAG_MASK: u64 = u64::MAX;

            let req = self
                .worker()
                .tag_recv(recvbuf, SCATTERV_TAG, TAG_MASK, &tag_param);
            if let Ok(Some(req)) = req {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }
    }

    /// Allgatherv: all ranks send `msg_size` bytes; every rank receives into variable-count slots.
    /// TODO(openshmem): variable-count collectives are not exposed by the current coll API.
    pub fn allgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        counts: &[usize],
        displs: &[usize],
    ) {
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
            return;
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLGATHERV_TAG: u64 = 0xBADC0DE8;
        const TAG_MASK: u64 = u64::MAX;

        let my_offset = displs[rank];
        let my_len = counts[rank];
        if my_offset + my_len <= recvbuf.len() {
            recvbuf[my_offset..my_offset + my_len].copy_from_slice(sendbuf);
        }

        // Phase 1: post all recvs
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut temp_bufs: Vec<Vec<u8>> = Vec::with_capacity(size);
        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                temp_bufs.push(Vec::new());
                continue;
            }
            let len = counts[peer];
            let mut buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut buf, ALLGATHERV_TAG, TAG_MASK, &tag_param);
            temp_bufs.push(buf);
            match req {
                Ok(r) => recv_reqs.push(r),
                _ => recv_reqs.push(None),
            }
        }

        // Phase 2: send our data to all peers
        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let req = self
                .endpoint(peer)
                .tag_send(sendbuf, ALLGATHERV_TAG, &tag_param);
            if let Ok(Some(req)) = req {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }

        // Phase 3: wait for all recvs and copy to recvbuf
        for peer in 0..size {
            if peer == rank {
                continue;
            }
            if let Some(req) = recv_reqs[peer].take() {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                let offset = displs[peer];
                let len = counts[peer];
                if offset + len <= recvbuf.len() {
                    recvbuf[offset..offset + len].copy_from_slice(&temp_bufs[peer]);
                }
            }
        }
    }

    /// Alltoallv: each rank sends variable-size pieces to every peer.
    /// TODO(openshmem): variable-count collectives are not exposed by the current coll API.
    pub fn alltoallv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        sendcounts: &[usize],
        sdispls: &[usize],
        recvcounts: &[usize],
        rdispls: &[usize],
    ) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            for p in 0..size {
                let src_off = sdispls[p];
                let dst_off = rdispls[p];
                let len = sendcounts[p];
                if src_off + len <= sendbuf.len() && dst_off + len <= recvbuf.len() {
                    recvbuf[dst_off..dst_off + len]
                        .copy_from_slice(&sendbuf[src_off..src_off + len]);
                }
            }
            return;
        }

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const ALLTOALLV_TAG: u64 = 0xBADC0DEA;
        const TAG_MASK: u64 = u64::MAX;

        // Phase 1: post recvs from all peers
        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        let mut temp_bufs: Vec<Vec<u8>> = Vec::with_capacity(size);
        for peer in 0..size {
            if peer == rank {
                let src_off = sdispls[peer];
                let dst_off = rdispls[peer];
                let len = sendcounts[peer];
                if src_off + len <= sendbuf.len() && dst_off + len <= recvbuf.len() {
                    recvbuf[dst_off..dst_off + len]
                        .copy_from_slice(&sendbuf[src_off..src_off + len]);
                }
                recv_reqs.push(None);
                temp_bufs.push(Vec::new());
                continue;
            }
            let len = recvcounts[peer];
            let mut buf = vec![0u8; len];
            match self
                .worker()
                .tag_recv(&mut buf, ALLTOALLV_TAG, TAG_MASK, &tag_param)
            {
                Ok(r) => {
                    temp_bufs.push(buf);
                    recv_reqs.push(r);
                }
                _ => {
                    temp_bufs.push(buf);
                    recv_reqs.push(None);
                }
            }
        }

        // Phase 2: send our piece to each peer
        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let src_off = sdispls[peer];
            let len = sendcounts[peer];
            if src_off + len <= sendbuf.len() {
                let req = self.endpoint(peer).tag_send(
                    &sendbuf[src_off..src_off + len],
                    ALLTOALLV_TAG,
                    &tag_param,
                );
                if let Ok(Some(req)) = req {
                    while !req.check_finished().unwrap_or(false) {
                        self.progress();
                    }
                }
            }
        }

        // Phase 3: wait for all recvs and copy to recvbuf
        for peer in 0..size {
            if peer == rank {
                continue;
            }
            if let Some(req) = recv_reqs[peer].take() {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
                let dst_off = rdispls[peer];
                let len = recvcounts[peer];
                if dst_off + len <= recvbuf.len() {
                    recvbuf[dst_off..dst_off + len].copy_from_slice(&temp_bufs[peer]);
                }
            }
        }
    }

    /// Alltoallw: like alltoallv but with per-peer datatypes.
    /// Since we only use bytes, this is identical to alltoallv.
    /// TODO(openshmem): `alltoallw` is not exposed by the current coll API.
    pub fn alltoallw(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        sendcounts: &[usize],
        sdispls: &[usize],
        recvcounts: &[usize],
        rdispls: &[usize],
    ) {
        self.alltoallv(sendbuf, recvbuf, sendcounts, sdispls, recvcounts, rdispls)
    }

    /// Neighbor allgather (ring topology).
    /// Each rank sends msg_size bytes to its two neighbors and receives from both.
    pub fn neighbor_allgather(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
            if copy_len > 0 {
                recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return;
        }

        let neighbors = [
            (rank.wrapping_sub(1)).rem_euclid(size),
            (rank + 1).rem_euclid(size),
        ];
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const TAG: u64 = 0xBADC0DEA;
        const TAG_MASK: u64 = u64::MAX;

        // Send to both neighbors
        for &dst in &neighbors {
            self.endpoint(dst)
                .tag_send(&sendbuf[..msg_size.min(sendbuf.len())], TAG, &tag_param)
                .expect("neighbor_allgather send");
        }

        // Receive from both neighbors
        for (neighbor_idx, &src) in neighbors.iter().enumerate() {
            let _ = src;
            let offset = neighbor_idx * msg_size;
            let end = (offset + msg_size).min(recvbuf.len());
            let mut recv_buf = vec![0u8; msg_size];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, TAG, TAG_MASK, &tag_param)
                .expect("neighbor_allgather recv")
                .expect("neighbor_allgather recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
            let copy_len = (end - offset).min(recv_buf.len());
            recvbuf[offset..offset + copy_len].copy_from_slice(&recv_buf[..copy_len]);
        }
    }

    /// Neighbor allgatherv (ring topology).
    /// Same as neighbor_allgather but with variable recv counts/displacements.
    pub fn neighbor_allgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_count: usize,
        recv_counts: &[usize],
        recv_displs: &[usize],
    ) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let offset = recv_displs.get(rank).copied().unwrap_or(0);
            let len = recv_counts.get(rank).copied().unwrap_or(0);
            let copy_len = len
                .min(sendbuf.len())
                .min(recvbuf.len().saturating_sub(offset));
            if copy_len > 0 {
                recvbuf[offset..offset + copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return;
        }

        let neighbors = [
            (rank.wrapping_sub(1)).rem_euclid(size),
            (rank + 1).rem_euclid(size),
        ];
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const TAG: u64 = 0xBADC0DEB;
        const TAG_MASK: u64 = u64::MAX;

        // Send to both neighbors
        for &dst in &neighbors {
            let len = send_count.min(sendbuf.len());
            self.endpoint(dst)
                .tag_send(&sendbuf[..len], TAG, &tag_param)
                .expect("neighbor_allgatherv send");
        }

        // Receive from both neighbors with variable counts/displacements
        for (neighbor_idx, _src) in neighbors.iter().enumerate() {
            let len = recv_counts.get(neighbor_idx).copied().unwrap_or(0);
            let offset = recv_displs.get(neighbor_idx).copied().unwrap_or(0);
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, TAG, TAG_MASK, &tag_param)
                .expect("neighbor_allgatherv recv")
                .expect("neighbor_allgatherv recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
            let copy_len = len.min(recvbuf.len().saturating_sub(offset));
            if copy_len > 0 {
                recvbuf[offset..offset + copy_len].copy_from_slice(&recv_buf[..copy_len]);
            }
        }
    }

    /// Neighbor alltoall (ring topology).
    /// Each rank sends msg_size bytes to each neighbor and receives msg_size bytes from each neighbor.
    pub fn neighbor_alltoall(&self, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let copy_len = msg_size.min(sendbuf.len()).min(recvbuf.len());
            if copy_len > 0 {
                recvbuf[..copy_len].copy_from_slice(&sendbuf[..copy_len]);
            }
            return;
        }

        let neighbors = [
            (rank.wrapping_sub(1)).rem_euclid(size),
            (rank + 1).rem_euclid(size),
        ];
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const TAG: u64 = 0xBADC0DEC;
        const TAG_MASK: u64 = u64::MAX;

        // Send to both neighbors (sendbuf[neighbor_idx * msg_size] goes to that neighbor)
        for (neighbor_idx, &dst) in neighbors.iter().enumerate() {
            let offset = neighbor_idx * msg_size;
            let end = (offset + msg_size).min(sendbuf.len());
            if offset < sendbuf.len() {
                self.endpoint(dst)
                    .tag_send(&sendbuf[offset..end], TAG, &tag_param)
                    .expect("neighbor_alltoall send");
            }
        }

        // Receive from both neighbors
        for (neighbor_idx, _src) in neighbors.iter().enumerate() {
            let offset = neighbor_idx * msg_size;
            let end = (offset + msg_size).min(recvbuf.len());
            let mut recv_buf = vec![0u8; msg_size];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, TAG, TAG_MASK, &tag_param)
                .expect("neighbor_alltoall recv")
                .expect("neighbor_alltoall recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
            let copy_len = (end - offset).min(recv_buf.len());
            recvbuf[offset..offset + copy_len].copy_from_slice(&recv_buf[..copy_len]);
        }
    }

    /// Neighbor alltoallv (ring topology).
    /// Variable counts/displacements for both send and receive.
    pub fn neighbor_alltoallv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        send_displs: &[usize],
        recv_counts: &[usize],
        recv_displs: &[usize],
    ) {
        let rank = self.rank;
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
            return;
        }

        let neighbors = [
            (rank.wrapping_sub(1)).rem_euclid(size),
            (rank + 1).rem_euclid(size),
        ];
        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const TAG: u64 = 0xBADC0DED;
        const TAG_MASK: u64 = u64::MAX;

        // Send to both neighbors using variable counts/displacements
        for (neighbor_idx, &dst) in neighbors.iter().enumerate() {
            let src_off = send_displs.get(neighbor_idx).copied().unwrap_or(0);
            let len = send_counts.get(neighbor_idx).copied().unwrap_or(0);
            let end = (src_off + len).min(sendbuf.len());
            if src_off < sendbuf.len() && len > 0 {
                self.endpoint(dst)
                    .tag_send(&sendbuf[src_off..end], TAG, &tag_param)
                    .expect("neighbor_alltoallv send");
            }
        }

        // Receive from both neighbors using variable counts/displacements
        for (neighbor_idx, _src) in neighbors.iter().enumerate() {
            let len = recv_counts.get(neighbor_idx).copied().unwrap_or(0);
            let offset = recv_displs.get(neighbor_idx).copied().unwrap_or(0);
            let mut recv_buf = vec![0u8; len];
            let req = self
                .worker()
                .tag_recv(&mut recv_buf, TAG, TAG_MASK, &tag_param)
                .expect("neighbor_alltoallv recv")
                .expect("neighbor_alltoallv recv request");
            while !req.check_finished().unwrap_or(false) {
                self.progress();
            }
            let copy_len = len.min(recvbuf.len().saturating_sub(offset));
            if copy_len > 0 {
                recvbuf[offset..offset + copy_len].copy_from_slice(&recv_buf[..copy_len]);
            }
        }
    }

    /// Neighbor alltoallw (ring topology).
    /// Same as alltoallv since we only use bytes.
    pub fn neighbor_alltoallw(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        send_counts: &[usize],
        send_displs: &[usize],
        recv_counts: &[usize],
        recv_displs: &[usize],
    ) {
        self.neighbor_alltoallv(
            sendbuf,
            recvbuf,
            send_counts,
            send_displs,
            recv_counts,
            recv_displs,
        )
    }

    /// Reduce_scatter_block: all ranks send `elemcount` bytes; each rank receives `elemcount / numprocs` bytes.
    /// TODO(openshmem): reduce-scatter is not exposed by the current coll API.
    pub fn reduce_scatter_block(&self, sendbuf: &[u8], recvbuf: &mut [u8], elemcount: usize) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            let len = elemcount.min(sendbuf.len()).min(recvbuf.len());
            if len > 0 {
                recvbuf[..len].copy_from_slice(&sendbuf[..len]);
            }
            return;
        }

        let per_rank = elemcount / size;

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCE_SCATTER_BLOCK_TAG: u64 = 0xBADC0DEB;
        const TAG_MASK: u64 = u64::MAX;

        let mut gathered = vec![0u8; elemcount * size];
        gathered[rank * elemcount..(rank + 1) * elemcount].copy_from_slice(sendbuf);

        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }
            let dst = &mut gathered[peer * elemcount..(peer + 1) * elemcount];
            match self
                .worker()
                .tag_recv(dst, REDUCE_SCATTER_BLOCK_TAG, TAG_MASK, &tag_param)
            {
                Ok(r) => recv_reqs.push(r),
                _ => recv_reqs.push(None),
            }
        }

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let req = self
                .endpoint(peer)
                .tag_send(sendbuf, REDUCE_SCATTER_BLOCK_TAG, &tag_param);
            if let Ok(Some(req)) = req {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            if let Some(req) = recv_reqs[peer].take() {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }

        let my_offset = rank * per_rank;
        for i in 0..per_rank {
            let mut sum: u64 = 0;
            for peer in 0..size {
                sum += gathered[peer * elemcount + my_offset + i] as u64;
            }
            recvbuf[i] = (sum % 256) as u8;
        }
    }

    /// Reduce-scatter: all ranks send `total` bytes; each rank receives `counts[rank]` bytes.
    /// TODO(openshmem): reduce-scatter is not exposed by the current coll API.
    pub fn reducescatter(&self, sendbuf: &[u8], recvbuf: &mut [u8], counts: &[usize]) {
        let rank = self.rank;
        let size = self.size;

        if size <= 1 {
            if !sendbuf.is_empty() && !recvbuf.is_empty() {
                let len = counts[rank].min(sendbuf.len()).min(recvbuf.len());
                recvbuf[..len].copy_from_slice(&sendbuf[..len]);
            }
            return;
        }

        let mut displs: Vec<usize> = Vec::with_capacity(size);
        let mut d = 0;
        for &c in counts {
            displs.push(d);
            d += c;
        }
        let total = d;

        let tag_param = RequestParamBuilder::new().no_imm_cmpl().build();
        const REDUCESCATTER_TAG: u64 = 0xBADC0DE9;
        const TAG_MASK: u64 = u64::MAX;

        let mut gathered = vec![0u8; total * size];
        gathered[rank * total..(rank + 1) * total].copy_from_slice(sendbuf);

        let mut recv_reqs: Vec<Option<ucx_sys::Request>> = Vec::with_capacity(size);
        for peer in 0..size {
            if peer == rank {
                recv_reqs.push(None);
                continue;
            }
            let dst = &mut gathered[peer * total..(peer + 1) * total];
            match self
                .worker()
                .tag_recv(dst, REDUCESCATTER_TAG, TAG_MASK, &tag_param)
            {
                Ok(r) => recv_reqs.push(r),
                _ => recv_reqs.push(None),
            }
        }

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            let req = self
                .endpoint(peer)
                .tag_send(sendbuf, REDUCESCATTER_TAG, &tag_param);
            if let Ok(Some(req)) = req {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }

        for peer in 0..size {
            if peer == rank {
                continue;
            }
            if let Some(req) = recv_reqs[peer].take() {
                while !req.check_finished().unwrap_or(false) {
                    self.progress();
                }
            }
        }

        let my_offset = displs[rank];
        let my_len = counts[rank];
        for i in 0..my_len {
            let mut sum: u64 = 0;
            for peer in 0..size {
                sum += gathered[peer * total + my_offset + i] as u64;
            }
            recvbuf[i] = (sum % 256) as u8;
        }
    }
}
