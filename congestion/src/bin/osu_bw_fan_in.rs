//! OSU MPI Bandwidth Fan-in Test (v7.5.2)
//!
//! Measures bandwidth of fan-in communication pattern across multiple nodes.
//! Non-parent nodes send data to the parent node in a windowed pattern.
//!
//! Protocol (matches C reference osu_bw_fan_in.c):
//! - Each non-parent node: sends `window_size` messages → receives 1-byte ACK
//! - Parent node: receives from all non-parent nodes → sends 1-byte ACK to each
//!
//! Requires at least 2 nodes with equal PPn (processes per node).
//! On single-node runs, prints warning and exits gracefully.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 4 ./target/release/osu_bw_fan_in
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use osu_congestion::fan_util::{ACK_TAG, DATA_TAG, TAG_MASK, fan_init, progress_worker};
use std::io;
use std::process;

/// Run the fan-in bandwidth benchmark.
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

    // Barrier so all nodes are ready
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
            if !is_parent {
                run_fan_in_send(
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
                run_fan_in_recv(
                    ctx,
                    worker,
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

            if !is_parent {
                run_fan_in_send(
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
                run_fan_in_recv(
                    ctx,
                    worker,
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
        // and by (total_nodes - 1) (number of child nodes)
        let normalized_time = t_total_reduced / ppn as f64 / (total_nodes - 1) as f64;

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

/// Non-parent side of fan-in: send window_size messages, receive ACK.
#[allow(clippy::too_many_arguments)]
fn run_fan_in_send(
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
    let parent_rank = ranks_queue[0];
    let ep = ctx.endpoint(parent_rank);

    // Send window_size messages to parent
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

    // Receive ACK from parent
    let ack_req = worker
        .tag_recv(ack_recv, ACK_TAG, TAG_MASK, tag_param)
        .expect("ack_recv")
        .expect("ack request");
    while !ack_req.check_finished().unwrap_or(false) {
        progress_worker(worker);
    }
}

/// Parent side of fan-in: receive from each child, send ACK.
#[allow(clippy::too_many_arguments)]
fn run_fan_in_recv(
    ctx: &OsUContext,
    worker: &ucx_sys::worker::Worker,
    recv_buf: &mut [u8],
    ack_send: &[u8],
    _ack_recv: &mut [u8],
    ranks_queue: &[usize],
    window_size: usize,
    tag_param: &ucx_sys::RequestParam,
) {
    for &child_rank in ranks_queue {
        // Receive window_size messages from child
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

        // Send ACK back to child
        ctx.endpoint(child_rank)
            .tag_send(ack_send, ACK_TAG, tag_param)
            .expect("ack_send");
    }
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
        output::print_header(&mut out, "Bandwidth Fan-in", BenchmarkType::CongestionBw);
        output::print_bandwidth_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCX → PMIx in order
}
