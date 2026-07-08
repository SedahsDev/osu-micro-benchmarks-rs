//! Shared fan-in/fan-out topology initialization utilities.
//!
//! Mirrors the C reference `osu_bw_fan_util.c` / `osu_bw_fan_util.h` from
//! OSU Micro-Benchmarks v7.5.2. Provides fan topology discovery via
//! hostname allgather, node ID assignment, and ranks queue construction.
//!
//! Both `osu_bw_fan_in` and `osu_bw_fan_out` benchmarks share this module
//! to avoid duplicating the topology discovery logic.

use osu_common::runtime::OsUContext;
use std::process;

/// UCX tag for data messages (matches C tag 100).
pub const DATA_TAG: u64 = 100;
/// UCX tag for ACK messages (matches C tag 101).
pub const ACK_TAG: u64 = 101;
/// Tag mask for receiving any tag.
pub const TAG_MASK: u64 = u64::MAX;
/// Max processor name length (matches MPI_MAX_PROCESSOR_NAME = 256).
pub const MAX_PROC_NAME_LEN: usize = 256;

/// Fan-in/out topology information.
///
/// After `fan_init()`, each process knows:
/// - Whether it is the parent node or a child node
/// - The ranks it needs to communicate with (parent rank or child ranks)
/// - The number of nodes and processes-per-node (PPN)
#[derive(Debug)]
pub struct FanTopology {
    pub total_nodes: usize,
    pub ppn: usize,
    pub is_parent: bool,
    /// Ranks queue — for non-parent: `[parent_rank]`.
    /// For parent: `[child_rank_0, child_rank_1, ...]`.
    pub ranks_queue: Vec<usize>,
}

/// Initialize fan-in/out topology by exchanging processor names via allgather.
///
/// Algorithm:
/// 1. Allgather hostnames from all processes
/// 2. Rank 0 sorts (hostname, rank) pairs and assigns node IDs + local ranks
/// 3. Validates equal PPN across all nodes
/// 4. Broadcasts topology to all processes
/// 5. Each process determines parent/child role and builds ranks queue
///
/// On single-node runs, produces a simulated 2-node topology with a warning.
pub fn fan_init(ctx: &OsUContext) -> FanTopology {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("[fan_util] Error: Need at least 2 processes");
        process::exit(1);
    }

    let hostname = get_hostname();
    let hostname_bytes = hostname.as_bytes();

    // Gather all hostnames via allgather
    let mut all_hostnames_raw = vec![0u8; size * MAX_PROC_NAME_LEN];
    let mut my_hostname_padded = vec![0u8; MAX_PROC_NAME_LEN];
    let copy_len = hostname_bytes.len().min(MAX_PROC_NAME_LEN);
    my_hostname_padded[..copy_len].copy_from_slice(&hostname_bytes[..copy_len]);

    ctx.allgather(
        &my_hostname_padded,
        &mut all_hostnames_raw,
        MAX_PROC_NAME_LEN,
    );

    // Parse hostnames
    let mut hostnames: Vec<String> = Vec::with_capacity(size);
    for i in 0..size {
        let offset = i * MAX_PROC_NAME_LEN;
        let end = offset + MAX_PROC_NAME_LEN;
        let bytes = &all_hostnames_raw[offset..end];
        let len = bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_PROC_NAME_LEN);
        hostnames.push(String::from_utf8_lossy(&bytes[..len]).to_string());
    }

    // Compute topology on rank 0, then broadcast
    let mut node_id_map = vec![0usize; size];
    let mut local_rank_map = vec![0usize; size];
    let mut total_nodes: usize = 1;
    let mut ppn: usize = 0;

    if rank == 0 {
        // Build (hostname, rank) pairs and sort by hostname
        let mut rank_by_name: Vec<(String, usize)> = hostnames
            .iter()
            .enumerate()
            .map(|(r, h)| (h.clone(), r))
            .collect();
        rank_by_name.sort_by(|a, b| a.0.cmp(&b.0));

        // Assign node IDs and local ranks
        let mut num_nodes = 1;
        let mut local_rank_counter: usize = 0;
        let mut first_ppn: Option<usize> = None;

        for (idx, (_name, _rank)) in rank_by_name.iter().enumerate() {
            if idx > 0 && rank_by_name[idx].0 != rank_by_name[idx - 1].0 {
                if let Some(fp) = first_ppn {
                    if fp != local_rank_counter {
                        eprintln!(
                            "[fan_util] Error: Please run with same ppn for all nodes \
                             (got {} and {})",
                            fp, local_rank_counter
                        );
                        process::exit(1);
                    }
                } else {
                    first_ppn = Some(local_rank_counter);
                }
                num_nodes += 1;
                local_rank_counter = 0;
            }
            let orig_rank = rank_by_name[idx].1;
            node_id_map[orig_rank] = num_nodes;
            local_rank_map[orig_rank] = local_rank_counter;
            local_rank_counter += 1;
        }

        ppn = first_ppn.unwrap_or(local_rank_counter);
        total_nodes = num_nodes;

        // Handle single-node case
        if total_nodes == 1 {
            eprintln!(
                "[fan_util] Warning: Running on single node — \
                 fan-in/out benchmarks require at least 2 nodes. \
                 Producing placeholder output."
            );
            // Simulate: rank 0 = parent (node 1), all others = children (node 2)
            total_nodes = 2;
            ppn = 1;
            for r in 1..size {
                node_id_map[r] = 2;
                local_rank_map[r] = 0;
            }
            node_id_map[0] = 1;
            local_rank_map[0] = 0;
        }
    }

    // Broadcast node_id_map from rank 0
    let mut node_id_send = vec![0u8; size * 4];
    let mut node_id_recv = vec![0u8; size * 4];
    if rank == 0 {
        for i in 0..size {
            node_id_send[i * 4..(i + 1) * 4].copy_from_slice(&node_id_map[i].to_le_bytes());
        }
    }
    ctx.bcast(&node_id_send, &mut node_id_recv, 0);
    node_id_map.copy_from_slice(
        &node_id_recv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect::<Vec<_>>(),
    );

    // Broadcast local_rank_map from rank 0
    let mut lrank_send = vec![0u8; size * 4];
    let mut lrank_recv = vec![0u8; size * 4];
    if rank == 0 {
        for i in 0..size {
            lrank_send[i * 4..(i + 1) * 4].copy_from_slice(&local_rank_map[i].to_le_bytes());
        }
    }
    ctx.bcast(&lrank_send, &mut lrank_recv, 0);
    local_rank_map.copy_from_slice(
        &lrank_recv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect::<Vec<_>>(),
    );

    // Broadcast total_nodes
    let tn_send = (total_nodes as u32).to_le_bytes();
    let mut tn_recv = [0u8; 4];
    ctx.bcast(&tn_send, &mut tn_recv, 0);
    total_nodes = u32::from_le_bytes(tn_recv) as usize;

    // Broadcast ppn
    let ppn_send = (ppn as u32).to_le_bytes();
    let mut ppn_recv = [0u8; 4];
    ctx.bcast(&ppn_send, &mut ppn_recv, 0);
    ppn = u32::from_le_bytes(ppn_recv) as usize;

    let my_node_id = node_id_map[rank];
    let _my_local_rank = local_rank_map[rank];
    let is_parent = my_node_id == 1;

    // Build ranks_queue
    let ranks_queue = if is_parent {
        // Parent: list of child ranks (local_rank==0 of each non-parent node)
        (0..size)
            .filter(|&r| node_id_map[r] >= 2 && local_rank_map[r] == 0)
            .collect()
    } else {
        // Non-parent: find parent rank (node_id==1, local_rank==0)
        let parent_rank = (0..size)
            .find(|&r| node_id_map[r] == 1 && local_rank_map[r] == 0)
            .unwrap_or(0);
        vec![parent_rank]
    };

    FanTopology {
        total_nodes,
        ppn,
        is_parent,
        ranks_queue,
    }
}

/// Get the local hostname.
pub fn get_hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::fs::read_to_string("/etc/hostname")
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "localhost".to_string())
    })
}

/// Progress the worker until no more work is available.
pub fn progress_worker(worker: &ucx_sys::worker::Worker) {
    loop {
        if !worker.progress() {
            break;
        }
    }
}
