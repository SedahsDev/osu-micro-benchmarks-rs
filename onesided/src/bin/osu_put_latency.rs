//! OSU MPI_Put Latency Test
//!
//! Measures one-sided Put latency using UCX RMA primitives.
//! Uses fence-based synchronization (flush) to match MPI_Win_fence semantics.
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_put_latency
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// Run the Put latency benchmark.
///
/// Rank 0 performs RMA Put to Rank 1's registered memory region.
/// Uses flush for fence-like synchronization.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size != 2 {
        if rank == 0 {
            eprintln!("Error: This test requires exactly 2 processes.");
        }
        process::exit(1);
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    let rma_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
    let flush_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Allocate send buffer
        let mut send_buf = vec![0u8; msg_size];
        // Fill with pattern for debugging
        for (i, byte) in send_buf.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        let remote_addr = ctx.remote_mem_addr(partner);
        let rkey = ctx
            .remote_rkey(partner)
            .expect("RMA context required for one-sided benchmark");

        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            if rank == 0 {
                // Sender: flush -> put -> flush (fence semantics)
                flush_blocking(worker, &flush_param);
                ep.rma_put(&send_buf, remote_addr, rkey, &rma_param)
                    .expect("rma_put");
                flush_blocking(worker, &flush_param);
            } else {
                // Target: flush -> flush (fence on both sides)
                flush_blocking(worker, &flush_param);
                flush_blocking(worker, &flush_param);
            }
        }

        // Timed iterations
        for _ in 0..iterations {
            let start = Wtime::new();
            if rank == 0 {
                flush_blocking(worker, &flush_param);
                ep.rma_put(&send_buf, remote_addr, rkey, &rma_param)
                    .expect("rma_put");
                flush_blocking(worker, &flush_param);
            } else {
                flush_blocking(worker, &flush_param);
                flush_blocking(worker, &flush_param);
            }
            let elapsed = start.elapsed_us();
            result.add(elapsed);
        }

        if rank == 0 {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_latency_row(
                &mut out,
                msg_size,
                result.avg_us,
                result.min_us,
                result.max_us,
            );
            output::print_newline(&mut out);
        }
    }
}

/// Flush the worker until all outstanding operations complete.
fn flush_blocking(worker: &ucx_sys::worker::Worker, param: &ucx_sys::RequestParam) {
    let req = worker.flush(param).expect("flush");
    if let Some(r) = req {
        while !r.check_finished().unwrap_or(false) {
            loop {
                if !worker.progress() {
                    break;
                }
            }
        }
    }
}

fn main() {
    let args = CliArgs::parse();

    // Allocate a target buffer for RMA registration
    // The buffer needs to be large enough for the max message size
    let max_size = args.max_message_size;
    let mut rma_target = vec![0u8; max_size];

    // Create the unified runtime context with RMA support
    let ctx = OsUContext::init_with_rma(args.ucc_backend(), Some(&mut rma_target));

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "MPI_Put Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
