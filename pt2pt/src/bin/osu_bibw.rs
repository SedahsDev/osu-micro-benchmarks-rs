//! OSU MPI Bi-Directional Bandwidth Test (v7.5.2)
//!
//! Measures bidirectional point-to-point bandwidth using UCX tag matching.
//! Both ranks simultaneously send and receive a window of messages.
//!
//! Protocol (matches C reference exactly):
//! - Rank 0: posts window of Ireceivs from rank 1 (tag=10) → posts window of Isends to rank 1 (tag=100) → waits all sends → waits all receives
//! - Rank 1: posts window of Ireceivs from rank 0 (tag=100) → posts window of Isends to rank 0 (tag=10) → waits all receives → waits all sends
//!
//! Bandwidth = (msg_size * iterations * window_size * 2) / total_time / 1e6  (MB/s)
//! The "* 2" accounts for both directions of traffic.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_bibw
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// UCX tag for rank 0 → rank 1 data messages.
const TAG_0_TO_1: u64 = 100;
/// UCX tag for rank 1 → rank 0 data messages.
const TAG_1_TO_0: u64 = 10;

/// Run the bidirectional bandwidth benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires exactly two processes.");
        process::exit(1);
    }
    if rank >= 2 {
        return;
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
    let window_size = args.window_size;

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let send_buf = vec![0u8; msg_size];
        let mut recv_buf = vec![0u8; msg_size];

        // Barrier before timing
        ctx.barrier();

        // Warmup
        for _ in 0..skip {
            do_bibw_iteration(
                rank,
                ep,
                worker,
                &tag_param,
                &send_buf,
                &mut recv_buf,
                window_size,
            );
        }

        // Timed iterations
        let start = Wtime::new();
        for _ in 0..iterations {
            do_bibw_iteration(
                rank,
                ep,
                worker,
                &tag_param,
                &send_buf,
                &mut recv_buf,
                window_size,
            );
        }
        let elapsed_us = start.elapsed_us();

        if rank == 0 {
            // Bandwidth = (msg_size * iterations * window_size * 2) / total_time / 1e6 (MB/s)
            let total_bytes = msg_size as f64 * iterations as f64 * window_size as f64 * 2.0;
            let total_time_s = elapsed_us / 1_000_000.0;
            let bandwidth_mbps = total_bytes / total_time_s / 1_000_000.0;

            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_bandwidth_avg(&mut out, msg_size, bandwidth_mbps);
            output::print_newline(&mut out);
        }
    }
}

/// Single bidirectional bandwidth iteration.
///
/// Both ranks post receives first, then sends, then wait for all.
/// This matches the C reference pattern exactly:
/// - Rank 0: recv from rank 1 (tag=10), send to rank 1 (tag=100)
/// - Rank 1: recv from rank 0 (tag=100), send to rank 0 (tag=10)
fn do_bibw_iteration(
    rank: usize,
    ep: &ucx_sys::ep::Ep,
    worker: &ucx_sys::worker::Worker,
    tag_param: &ucx_sys::RequestParam,
    send_buf: &[u8],
    recv_buf: &mut [u8],
    window_size: usize,
) {
    let recv_tag = if rank == 0 { TAG_1_TO_0 } else { TAG_0_TO_1 };
    let send_tag = if rank == 0 { TAG_0_TO_1 } else { TAG_1_TO_0 };

    // Post window of receives
    let mut recv_reqs: Vec<ucx_sys::Request> = Vec::with_capacity(window_size);
    for _ in 0..window_size {
        let req = worker
            .tag_recv(recv_buf, recv_tag, u64::MAX, tag_param)
            .expect("tag_recv")
            .expect("recv request");
        recv_reqs.push(req);
    }

    // Post window of sends
    let mut send_reqs: Vec<ucx_sys::Request> = Vec::with_capacity(window_size);
    for _ in 0..window_size {
        if let Some(r) = ep
            .tag_send(send_buf, send_tag, tag_param)
            .expect("tag_send")
        {
            send_reqs.push(r);
        }
    }

    // Wait for all sends
    for req in &send_reqs {
        while !req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
    }

    // Wait for all receives
    for req in &recv_reqs {
        while !req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
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

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init(args.ucc_backend());

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Bi-Directional Bandwidth",
            BenchmarkType::BiBandwidth,
        );
        output::print_bandwidth_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
