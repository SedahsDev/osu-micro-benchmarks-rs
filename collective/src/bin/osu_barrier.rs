//! OSU MPI Barrier Latency Test (v7.5.2)
//!
//! Measures collective barrier latency using UCX tag-matching barrier fallback.
//!
//! Requires at least 2 processes. Repeatedly calls barrier and reports
//! min/avg/max latency.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_barrier
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    let iterations = args.iterations;
    let skip = args.skip;

    // Barrier so all processes are ready
    ctx.barrier();

    let mut timer: f64 = 0.0;

    for i in 0..(iterations + skip) {
        if i == skip {
            // Warmup done, start timing
        }

        let t_start = Wtime::new();

        // UCX tag-matching barrier
        ctx.barrier();

        let elapsed_us = t_start.elapsed_us();

        if i >= skip {
            timer += elapsed_us;

            // Barrier after timed iteration to synchronize
            ctx.barrier();
        }
    }

    let latency = timer / iterations as f64;

    // Reduce to get min/max/avg across all ranks
    let min_time = ctx.allreduce_min_f64(latency);
    let max_time = ctx.allreduce_max_f64(latency);
    let sum_time = ctx.allreduce_sum_f64(latency);
    let avg_time = sum_time / size as f64;

    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_latency_row(&mut out, 0, avg_time, min_time, max_time);
        output::print_newline(&mut out);
    }
}

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init();

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Barrier", BenchmarkType::CollectiveLatency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCX → PMIx in order
}
