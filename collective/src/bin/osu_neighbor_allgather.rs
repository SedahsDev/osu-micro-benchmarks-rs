//! OSU MPI Neighborhood Allgather Latency Test
//!
//! Measures neighbor allgather latency using UCX tag-matching fallback.
//! Ring topology: each rank communicates with (rank-1)%size and (rank+1)%size.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_neighbor_allgather
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the neighbor allgather latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    // Allocate buffers — recvbuf needs 2 * max_message_size (2 neighbors)
    let max_buf_size = args.max_message_size;
    let mut sendbuf = vec![0u8; max_buf_size];
    let mut recvbuf = vec![0u8; max_buf_size * 2];

    // Barrier so all processes are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Initialize send buffer: each byte = rank + 1
        for item in sendbuf.iter_mut().take(msg_size) {
            *item = (rank + 1) as u8;
        }

        let send_slice = &sendbuf[..msg_size];
        let recv_slice = &mut recvbuf[..msg_size * 2];

        // Warmup iterations
        for _ in 0..skip {
            ctx.neighbor_allgather(send_slice, recv_slice, msg_size);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            ctx.neighbor_allgather(send_slice, recv_slice, msg_size);
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

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init(args.ucc_backend());

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Neighbor Allgather Latency",
            BenchmarkType::CollectiveLatency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
