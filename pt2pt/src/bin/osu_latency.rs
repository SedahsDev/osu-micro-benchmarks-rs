//! OSU MPI Latency Test (v7.5.2)
//!
//! Measures point-to-point message latency using UCX tag matching.
//!
//! Requires exactly 2 processes. Runs ping-pong Send/Recv for each
//! message size from min to max, reporting min/avg/max latency per size.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_latency
//! ```

use osu_common::cli::{CliArgs, LARGE_MESSAGE_SIZE};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// UCX tag used for all messages in this benchmark.
const TAG: u64 = 0x123456789ABCDEF0;
/// Tag mask for exact matching.
const TAG_MASK: u64 = u64::MAX;

/// Run the latency benchmark using UCX tag matching.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires at least 2 processes.");
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

    // Barrier so both sides are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let mut buf = vec![0u8; msg_size];
        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
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

        // Timed iterations
        for _ in 0..iterations {
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
            let elapsed = start.elapsed_us();
            result.add(elapsed);
        }

        if rank == 0 {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_latency_row(&mut out, msg_size, result.avg_us, result.min_us, result.max_us);
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
            // Use multiplication for large messages, addition for small
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
        output::print_header(&mut out, "Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCC → UCX → PMIx in order
}
