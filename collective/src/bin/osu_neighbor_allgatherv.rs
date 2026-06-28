//! OSU MPI Neighborhood Allgatherv Latency Test
//!
//! Measures neighbor allgatherv latency using UCX tag-matching fallback.
//! Ring topology: each rank communicates with (rank-1)%size and (rank+1)%size.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_neighbor_allgatherv
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the neighbor allgatherv latency benchmark.
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

    // For neighbor allgatherv with 2 neighbors, use equal counts/displacements
    let num_neighbors = 2;
    let mut recv_counts = vec![0usize; num_neighbors];
    let mut recv_displs = vec![0usize; num_neighbors];

    // Barrier so all processes are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Initialize send buffer: each byte = rank + 1
        for item in sendbuf.iter_mut().take(msg_size) {
            *item = (rank + 1) as u8;
        }

        // Set up recv counts and displacements for 2 neighbors
        for i in 0..num_neighbors {
            recv_counts[i] = msg_size;
            recv_displs[i] = i * msg_size;
        }

        let send_slice = &sendbuf[..msg_size];
        let recv_slice = &mut recvbuf[..msg_size * 2];

        // Warmup iterations
        for _ in 0..skip {
            ctx.neighbor_allgatherv(send_slice, recv_slice, msg_size, &recv_counts, &recv_displs);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            ctx.neighbor_allgatherv(send_slice, recv_slice, msg_size, &recv_counts, &recv_displs);
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

    let ctx = OsUContext::init(args.ucc_backend());

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Neighbor Allgatherv Latency",
            BenchmarkType::CollectiveLatency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
