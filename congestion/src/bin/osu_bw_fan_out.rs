//! OSU MPI Bandwidth Fan-out Test (v7.5.2)
//!
//! Measures bandwidth of fan-out communication pattern across multiple nodes.
//! Parent node sends data to all non-parent nodes in a windowed pattern.
//!
//! Protocol (matches C reference osu_bw_fan_out.c):
//! - Parent: sends `window_size` messages to each child → receives 1-byte ACK from each
//! - Each non-parent node: receives `window_size` messages → sends 1-byte ACK
//!
//! Requires at least 2 nodes with equal PPn (processes per node).
//! On single-node runs, prints warning and exits gracefully.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 4 ./target/release/osu_bw_fan_out
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// UCX tag for data messages (matches C tag 100).
const DATA_TAG: u64 = 100;
/// UCX tag for ACK messages (matches C tag 101).
const ACK_TAG: u64 = 101;
/// Tag mask for receiving any tag.
const TAG_MASK: u64 = u64::MAX;
/// Max processor name length (matches MPI_MAX_PROCESSOR_NAME = 256).
const MAX_PROC_NAME_LEN: usize = 256;

/// Fan-in/out topology information.
struct FanTopology {
    total_nodes: usize,
    ppn: usize,
    is_parent: bool,
    /// Ranks queue — for non-parent: [parent_rank].
    /// For parent: [child_rank_0, child_rank_1, ...].
    ranks_queue: Vec<usize>,
}

/// Initialize fan-in/out topology by exchanging processor names via allgather.
fn fan_init(ctx: &OsUContext) -> FanTopology {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("[osu_bw_fan_out] Error: Need at least 2 processes");
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
        let mut rank_by_name: Vec<(String, usize)> = hostnames
            .iter()
            .enumerate()
            .map(|(r, h)| (h.clone(), r))
            .collect();
        rank_by_name.sort_by(|a, b| a.0.cmp(&b.0));

        let mut num_nodes = 1;
        let mut local_rank_counter: usize = 0;
        let mut first_ppn: Option<usize> = None;

        for (idx, (_name, _rank)) in rank_by_name.iter().enumerate() {
            if idx > 0 && rank_by_name[idx].0 != rank_by_name[idx - 1].0 {
                if let Some(fp) = first_ppn {
                    if fp != local_rank_counter {
                        eprintln!(
                            "[osu_bw_fan_out] Error: Please run with same ppn for all nodes \
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

        if total_nodes == 1 {
            eprintln!(
                "[osu_bw_fan_out] Warning: Running on single node — \
                 fan-in/out benchmarks require at least 2 nodes. \
                 Producing placeholder output."
            );
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

    let ranks_queue = if is_parent {
        (0..size)
            .filter(|&r| node_id_map[r] >= 2 && local_rank_map[r] == 0)
            .collect()
    } else {
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
fn get_hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::fs::read_to_string("/etc/hostname")
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "localhost".to_string())
    })
}

/// Progress the worker until no more work is available.
fn progress_worker(worker: &ucx_sys::worker::Worker) {
    loop {
        if !worker.progress() {
            break;
        }
    }
}

/// Run the fan-out bandwidth benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires at least two processes.");
        process::exit(1);
    }

    let topology = fan_init(ctx);
    let is_parent = topology.is_parent;
    let ranks_queue = &topology.ranks_queue;
    let ppn = topology.ppn;
    let total_nodes = topology.total_nodes;

    let worker = ctx.worker();
    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    let window_size = args.window_size;

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let send_buf = vec![0u8; msg_size];
        let mut recv_buf = vec![0u8; msg_size];
        let ack_send = [0u8; 1];
        let mut ack_recv = [0u8; 1];

        ctx.barrier();

        // Skip warmup iterations
        for _ in 0..skip {
            if is_parent {
                run_fan_out_send(
                    ctx,
                    worker,
                    &send_buf,
                    &mut recv_buf,
                    &ack_send,
                    &mut ack_recv,
                    ranks_queue,
                    window_size,
                    &tag_param,
                );
            } else {
                run_fan_out_recv(
                    ctx,
                    worker,
                    &send_buf,
                    &mut recv_buf,
                    &ack_send,
                    &mut ack_recv,
                    ranks_queue,
                    window_size,
                    &tag_param,
                );
            }
        }

        // Timed iterations
        let mut t_total: f64 = 0.0;
        for _ in 0..iterations {
            let start = Wtime::new();

            if is_parent {
                run_fan_out_send(
                    ctx,
                    worker,
                    &send_buf,
                    &mut recv_buf,
                    &ack_send,
                    &mut ack_recv,
                    ranks_queue,
                    window_size,
                    &tag_param,
                );
            } else {
                run_fan_out_recv(
                    ctx,
                    worker,
                    &send_buf,
                    &mut recv_buf,
                    &ack_send,
                    &mut ack_recv,
                    ranks_queue,
                    window_size,
                    &tag_param,
                );
            }

            let elapsed_us = start.elapsed_us();
            t_total += elapsed_us;
        }

        // Reduce t_total across all processes (SUM), then normalize
        let t_total_reduced = ctx.allreduce_sum_f64(t_total);

        // Normalize: divide by ppn (aggregate across parents on same node)
        let normalized_time = t_total_reduced / ppn as f64;

        // Rank 0 prints results
        if rank == 0 {
            let total_bytes = msg_size as f64
                * ppn as f64
                * iterations as f64
                * window_size as f64
                * (total_nodes - 1) as f64;
            let total_time_s = normalized_time / 1_000_000.0;
            let bandwidth_mbps = total_bytes / total_time_s / 1_000_000.0;

            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_bandwidth_avg(&mut out, msg_size, bandwidth_mbps);
            output::print_newline(&mut out);
        }
    }
}

/// Parent side of fan-out: send to each child, receive ACK.
#[allow(clippy::too_many_arguments)]
fn run_fan_out_send(
    ctx: &OsUContext,
    worker: &ucx_sys::worker::Worker,
    send_buf: &[u8],
    _recv_buf: &mut [u8],
    _ack_send: &[u8],
    ack_recv: &mut [u8],
    ranks_queue: &[usize],
    window_size: usize,
    tag_param: &ucx_sys::RequestParam,
) {
    for &child_rank in ranks_queue {
        let ep = ctx.endpoint(child_rank);

        // Send window_size messages to child
        let mut send_reqs: Vec<ucx_sys::Request> = Vec::with_capacity(window_size);
        for _ in 0..window_size {
            let req = ep
                .tag_send(send_buf, DATA_TAG, tag_param)
                .expect("tag_send");
            if let Some(r) = req {
                send_reqs.push(r);
            }
        }
        for req in &send_reqs {
            while !req.check_finished().unwrap_or(false) {
                progress_worker(worker);
            }
        }

        // Receive ACK from child
        let ack_req = worker
            .tag_recv(ack_recv, ACK_TAG, TAG_MASK, tag_param)
            .expect("ack_recv")
            .expect("ack request");
        while !ack_req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
    }
}

/// Non-parent side of fan-out: receive from parent, send ACK.
#[allow(clippy::too_many_arguments)]
fn run_fan_out_recv(
    ctx: &OsUContext,
    worker: &ucx_sys::worker::Worker,
    _send_buf: &[u8],
    recv_buf: &mut [u8],
    ack_send: &[u8],
    _ack_recv: &mut [u8],
    ranks_queue: &[usize],
    window_size: usize,
    tag_param: &ucx_sys::RequestParam,
) {
    let parent_rank = ranks_queue[0];

    // Receive window_size messages from parent
    let mut recv_reqs: Vec<ucx_sys::Request> = Vec::with_capacity(window_size);
    for _ in 0..window_size {
        let req = worker
            .tag_recv(recv_buf, DATA_TAG, TAG_MASK, tag_param)
            .expect("tag_recv")
            .expect("recv request");
        recv_reqs.push(req);
    }
    for req in &recv_reqs {
        while !req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
    }

    // Send ACK to parent
    ctx.endpoint(parent_rank)
        .tag_send(ack_send, ACK_TAG, tag_param)
        .expect("ack_send");
}

/// Generate message sizes from min to max using the given increment.
fn message_sizes(args: &CliArgs) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut size = args.min_message_size;
    while size <= args.max_message_size {
        sizes.push(size);
        if size == 0 {
            size = 1;
        } else {
            if size <= osu_common::cli::LARGE_MESSAGE_SIZE {
                size *= args.message_size_incr;
            } else {
                size += args.message_size_incr;
            }
        }
    }
    sizes
}

fn main() {
    let args = CliArgs::parse();

    // Create the unified runtime context
    let ctx = OsUContext::init(args.ucc_backend());

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Bandwidth Fan-out", BenchmarkType::CongestionBw);
        output::print_bandwidth_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCX → PMIx in order
}
