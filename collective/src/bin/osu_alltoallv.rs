//! OSU MPI Alltoallv Latency Test (v7.5.2)
//!
//! Measures alltoallv latency using UCX tag-matching fallback.
//!
//! Requires at least 2 processes. Runs alltoallv with various message sizes,
//! reporting latency in microseconds. Each peer sends/receives `msg_size` bytes.

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the alltoallv latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    // Allocate buffers — send/recv need size * max_message_size
    let max_buf_size = args.max_message_size;
    let total_buf_size = max_buf_size * size;
    let mut sendbuf = vec![0u8; total_buf_size];
    let mut recvbuf = vec![0u8; total_buf_size];

    // Barrier so all processes are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Build uniform sendcounts, sdispls, recvcounts, rdispls
        let sendcounts: Vec<usize> = vec![msg_size; size];
        let sdispls: Vec<usize> = (0..size).map(|i| i * msg_size).collect();
        let recvcounts: Vec<usize> = vec![msg_size; size];
        let rdispls: Vec<usize> = (0..size).map(|i| i * msg_size).collect();

        // Initialize send buffer: piece for peer p has bytes = p + 1
        for peer in 0..size {
            for i in 0..msg_size {
                sendbuf[peer * msg_size + i] = (peer + 1) as u8;
            }
        }

        let send_slice = &sendbuf[..msg_size * size];
        let recv_slice = &mut recvbuf[..msg_size * size];

        // Warmup iterations
        for _ in 0..skip {
            ctx.alltoallv(send_slice, recv_slice, &sendcounts, &sdispls, &recvcounts, &rdispls);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            ctx.alltoallv(send_slice, recv_slice, &sendcounts, &sdispls, &recvcounts, &rdispls);
            let elapsed_us = t_start.elapsed_us();
            // Barrier after each iteration to synchronize
            ctx.barrier();

            timer += elapsed_us;
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
            output::print_latency_row(&mut out, msg_size, avg_time, min_time, max_time);
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
            size *= args.message_size_incr;
        }
    }
    sizes
}

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init();

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "All-to-Allv Personalized Exchange Latency", BenchmarkType::CollectiveLatency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
