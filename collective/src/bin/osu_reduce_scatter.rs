//! OSU MPI Reduce-scatter Latency Test (v7.5.2)
//!
//! Measures reduce-scatter latency using UCX tag-matching fallback.
//!
//! Requires at least 2 processes. All ranks send `msg_size` bytes;
//! each rank receives `msg_size / numprocs` bytes (element-wise SUM).

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the reduce-scatter latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    let mut sendbuf = vec![0u8; args.max_message_size];
    let recv_per_rank = args.max_message_size / size;
    let mut recvbuf = vec![0u8; recv_per_rank];

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // msg_size must be divisible by size for uniform counts
        if msg_size % size != 0 {
            continue;
        }

        let per_rank = msg_size / size;

        // Initialize send buffer: each byte = rank + 1
        for item in sendbuf.iter_mut().take(msg_size) {
            *item = (rank + 1) as u8;
        }

        // Build uniform recvcounts
        let recvcounts: Vec<usize> = vec![per_rank; size];

        let send_slice = &sendbuf[..msg_size];
        let recv_slice = &mut recvbuf[..per_rank];

        // Warmup iterations
        for _ in 0..skip {
            ctx.reducescatter(send_slice, recv_slice, &recvcounts);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            ctx.reducescatter(send_slice, recv_slice, &recvcounts);
            let elapsed_us = t_start.elapsed_us();
            ctx.barrier();
            timer += elapsed_us;
        }

        let latency = timer / iterations as f64;

        let min_time = ctx.allreduce_min_f64(latency);
        let max_time = ctx.allreduce_max_f64(latency);
        let sum_time = ctx.allreduce_sum_f64(latency);
        let avg_time = sum_time / size as f64;

        if rank == 0 {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_latency_row(&mut out, msg_size, avg_time, min_time, max_time);
            output::print_newline(&mut out);
        }
    }
}

fn main() {
    let args = CliArgs::parse();
    let ctx = OsUContext::init(args.ucc_backend());

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Reduce-scatter Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
