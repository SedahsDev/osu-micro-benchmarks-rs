//! OSU MPI Scatter Latency Test (v7.5.2)
//!
//! Measures scatter latency using UCX tag-matching fallback.
//!
//! Requires at least 2 processes. Root sends `msg_size * numprocs` bytes;
//! each rank receives `msg_size` bytes.

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the scatter latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    let max_buf_size = args.max_message_size * size;
    let mut sendbuf = vec![0u8; max_buf_size];
    let mut recvbuf = vec![0u8; args.max_message_size];

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Initialize send buffer on root
        if rank == 0 {
            for (i, item) in sendbuf.iter_mut().enumerate().take(msg_size * size) {
                *item = (i % 256) as u8;
            }
        }

        let send_slice = &sendbuf[..msg_size * size];
        let recv_slice = &mut recvbuf[..msg_size];

        // Warmup iterations
        for _ in 0..skip {
            ctx.scatter(send_slice, recv_slice, msg_size, 0);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            ctx.scatter(send_slice, recv_slice, msg_size, 0);
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

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Scatter Latency",
            BenchmarkType::CollectiveLatency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
