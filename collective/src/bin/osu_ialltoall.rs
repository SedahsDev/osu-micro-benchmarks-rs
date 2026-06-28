//! OSU MPI Non-blocking Alltoall Latency Test (v7.5.2)
//!
//! Measures non-blocking collective alltoall latency using UCX tag-matching
//! alltoall fallback. Reports Overlap, CPU, Communication, Wait, and Init
//! times in microseconds.
//!
//! Requires at least 2 processes. Repeatedly calls ialltoall + wait and reports
//! timing breakdown.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_ialltoall
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::{OsUContext, OsURequest};
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

    // Fixed message size for now
    let msg_size = 0;
    let sendbuf = vec![0u8; msg_size * size];
    let mut recvbuf = vec![0u8; msg_size * size];

    let mut timer: f64 = 0.0;
    let mut tcomp_total: f64 = 0.0;
    let mut wait_total: f64 = 0.0;
    let mut init_total: f64 = 0.0;

    for i in 0..(iterations + skip) {
        let t_start = Wtime::new();

        // Init time: start the non-blocking operation
        let init_start = Wtime::new();
        let mut request: OsURequest = ctx.ialltoall(&sendbuf, &mut recvbuf, msg_size);
        let init_time = init_start.elapsed_us();

        // CPU/compute time: do dummy work while collective is in flight
        let comp_start = Wtime::new();
        // No real dummy compute for ialltoall — just measure the overlap
        let comp_time = comp_start.elapsed_us();

        // Wait time: wait for the non-blocking operation to complete
        let wait_start = Wtime::new();
        request.wait();
        let wait_time = wait_start.elapsed_us();

        let elapsed_us = t_start.elapsed_us();

        if i >= skip {
            timer += elapsed_us;
            tcomp_total += comp_time;
            wait_total += wait_time;
            init_total += init_time;

            // Barrier after timed iteration to synchronize
            ctx.barrier();
        }
    }

    let overlap = timer / iterations as f64;
    let cpu_avg = tcomp_total / iterations as f64;
    let wait_avg = wait_total / iterations as f64;
    let init_avg = init_total / iterations as f64;

    // Communication = Overlap - Init (time spent after init, before wait completes)
    let comm_avg = overlap - init_avg;

    // Reduce to get min/max/avg across all ranks for the overlap column
    let _overlap_min = ctx.allreduce_min_f64(overlap);
    let _overlap_max = ctx.allreduce_max_f64(overlap);

    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_nbc_header(&mut out);
        output::print_nbc_row(
            &mut out, msg_size, overlap, cpu_avg, comm_avg, wait_avg, init_avg,
        );
        output::print_newline(&mut out);
    }
}

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init(args.ucc_backend());

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Non-blocking Alltoall",
            BenchmarkType::NonBlockingCollective,
        );
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCX → PMIx in order
}
