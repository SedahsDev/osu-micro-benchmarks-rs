//! OSU MPI Multi-process Latency Test (v7.5.2)
//!
//! Measures point-to-point message latency with a barrier before each
//! send/recv pair, simulating cold-start conditions.
//!
//! Requires exactly 2 processes. Barrier → Send/Recv for each message
//! size from min to max, reporting average latency per size.
//!
//! Difference from osu_latency: the barrier between iterations measures
//! worst-case latency when the connection is "cold" (no in-flight state).
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_latency_mp
//! ```

use osu_common::cli::{CliArgs, LARGE_MESSAGE_SIZE};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// UCX tag used for all messages in this benchmark.
const TAG: u64 = 0x123456789ABCDEF0;
/// Tag mask for exact matching.
const TAG_MASK: u64 = u64::MAX;

/// Run the multi-process latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires at least 2 processes.");
        process::exit(1);
    }
    if rank >= 2 {
        return;
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let mut buf = vec![0u8; msg_size];

        // Skip warmup iterations
        for _ in 0..skip {
            ctx.barrier();
            if rank == 0 {
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
                ctx.progress();
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
            } else {
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
            }
        }

        // Timed iterations — barrier before each exchange
        let total_start = Wtime::new();
        for _ in 0..iterations {
            ctx.barrier();
            let start = Wtime::new();
            if rank == 0 {
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
                ctx.progress();
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
            } else {
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
            }
            let _elapsed = start.elapsed_us();
        }
        let total_elapsed = total_start.elapsed_us();

        if rank == 0 {
            // Latency = total_time / (2 * iterations) — divide by 2 for one-way
            let latency_us = total_elapsed / (2.0 * iterations as f64);

            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_latency_avg(&mut out, msg_size, latency_us);
            output::print_newline(&mut out);
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

    let ctx = OsUContext::init();

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Multi-process Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
