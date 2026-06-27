//! OSU MPI Bandwidth Test (v7.5.2)
//!
//! Measures point-to-point message bandwidth using UCX tag matching with
//! a windowed send/recv pattern.
//!
//! Protocol (matches C reference exactly):
//! - Rank 0: posts `window_size` tag_sends → waits for all → receives 1-byte ACK
//! - Rank 1: posts `window_size` tag_recvs → waits for all → sends 1-byte ACK
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_bw
//! ```

use osu_common::cli::{CliArgs, LARGE_MESSAGE_SIZE};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// UCX tag used for data messages (matches C tag 100).
const DATA_TAG: u64 = 100;
/// UCX tag used for ACK messages (matches C tag 101).
const ACK_TAG: u64 = 101;

/// Run the bandwidth benchmark using UCX tag matching with windowed sends.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires exactly two processes.");
        process::exit(1);
    }
    if rank >= 2 {
        // Only ranks 0 and 1 participate.
        return;
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    // Build request params for tag send/recv
    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    // Window size controls how many messages are in flight at once.
    let window_size = args.window_size;

    // Barrier so both sides are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Allocate send/recv buffers
        let send_buf = vec![0u8; msg_size];
        let mut recv_buf = vec![0u8; msg_size];

        // ACK buffers (1 byte)
        let ack_send = [0u8; 1];
        let mut ack_recv = [0u8; 1];

        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            do_bw_iteration(
                rank,
                ep,
                worker,
                &tag_param,
                &send_buf,
                &mut recv_buf,
                &ack_send,
                &mut ack_recv,
                window_size,
            );
        }

        // Timed iterations
        for _ in 0..iterations {
            let start = Wtime::new();

            do_bw_iteration(
                rank,
                ep,
                worker,
                &tag_param,
                &send_buf,
                &mut recv_buf,
                &ack_send,
                &mut ack_recv,
                window_size,
            );

            let elapsed_us = start.elapsed_us();
            result.add(elapsed_us);
        }

        // Rank 0 prints results
        if rank == 0 {
            // Bandwidth = (msg_size * window_size * iterations) / total_time
            // Result in MB/s
            let total_bytes = msg_size as f64 * window_size as f64 * iterations as f64;
            let total_time_s = result.total_us / 1_000_000.0;
            let bandwidth_mbps = total_bytes / total_time_s / 1_000_000.0;

            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_bandwidth_avg(&mut out, msg_size, bandwidth_mbps);
            output::print_newline(&mut out);
        }
    }
}

/// Perform a single bandwidth iteration.
///
/// Rank 0: sends window_size messages, waits, receives ACK.
/// Rank 1: receives window_size messages, waits, sends ACK.
#[allow(clippy::too_many_arguments)]
fn do_bw_iteration(
    rank: usize,
    ep: &ucx_sys::ep::Ep,
    worker: &ucx_sys::worker::Worker,
    tag_param: &ucx_sys::RequestParam,
    send_buf: &[u8],
    recv_buf: &mut [u8],
    ack_send: &[u8],
    ack_recv: &mut [u8],
    window_size: usize,
) {
    if rank == 0 {
        // Sender: post window of sends
        let mut send_reqs: Vec<ucx_sys::Request> = Vec::new();
        for _ in 0..window_size {
            let req = ep
                .tag_send(send_buf, DATA_TAG, tag_param)
                .expect("tag_send");
            // Only track non-eager requests (eager = None means immediate completion)
            if let Some(r) = req {
                send_reqs.push(r);
            }
        }

        // Wait for all sends to complete
        for req in &send_reqs {
            while !req.check_finished().unwrap_or(false) {
                progress_worker(worker);
            }
        }

        // Receive 1-byte ACK from rank 1
        let ack_req = worker
            .tag_recv(ack_recv, ACK_TAG, u64::MAX, tag_param)
            .expect("ack_recv")
            .expect("ack request");
        while !ack_req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
    } else {
        // Receiver: post window of receives
        let mut recv_reqs: Vec<ucx_sys::Request> = Vec::new();
        for _ in 0..window_size {
            let req = worker
                .tag_recv(recv_buf, DATA_TAG, u64::MAX, tag_param)
                .expect("tag_recv")
                .expect("recv request");
            recv_reqs.push(req);
        }

        // Wait for all receives to complete
        for req in &recv_reqs {
            while !req.check_finished().unwrap_or(false) {
                progress_worker(worker);
            }
        }

        // Send 1-byte ACK back to rank 0
        ep.tag_send(ack_send, ACK_TAG, tag_param).expect("ack_send");
    }
}

/// Progress the worker until no more work is available.
fn progress_worker(worker: &ucx_sys::worker::Worker) {
    loop {
        if !worker.progress() {
            break;
        }
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
            // Use multiplication for small messages, addition for large
            if size <= LARGE_MESSAGE_SIZE {
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
    let ctx = OsUContext::init();

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Bandwidth", BenchmarkType::Bandwidth);
        output::print_bandwidth_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCX → PMIx in order
}
