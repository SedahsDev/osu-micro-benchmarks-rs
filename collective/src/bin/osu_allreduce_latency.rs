//! OSU MPI Allreduce Latency Test (v7.5.2)
//!
//! Measures allreduce latency using UCX tag-matching fallback.
//!
//! Requires at least 2 processes. Runs allreduce with various message sizes,
//! reporting latency in microseconds.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_allreduce_latency
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the allreduce latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    // Allocate the maximum buffer needed
    let max_buf_size = args.max_message_size;
    let mut sendbuf = vec![0u8; max_buf_size];
    let mut recvbuf = vec![0u8; max_buf_size];

    // Barrier so all processes are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Initialize send buffer: each byte = rank + 1 (for validation-like pattern)
        for item in sendbuf.iter_mut().take(msg_size) {
            *item = (rank + 1) as u8;
        }

        // Use the slice for this message size
        let send_slice = &mut sendbuf[..msg_size];
        let recv_slice = &mut recvbuf[..msg_size];

        // Warmup iterations
        for _ in 0..skip {
            allreduce_ucx_fallback(ctx, send_slice, recv_slice, msg_size);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            allreduce_ucx_fallback(ctx, send_slice, recv_slice, msg_size);
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

/// UCX-based allreduce using all-gather + local sum.
fn allreduce_ucx_fallback(ctx: &OsUContext, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size <= 1 {
        recvbuf.copy_from_slice(sendbuf);
        return;
    }

    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
    const ALLREDUCE_TAG: u64 = 0xBADDEF01;
    const TAG_MASK: u64 = u64::MAX;

    let total_size = msg_size * size;
    let mut gathered = vec![0u8; total_size];

    let my_offset = rank * msg_size;
    gathered[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);

    // Send our data to all peers
    for peer in 0..size {
        if peer != rank {
            ctx.endpoint(peer)
                .tag_send(sendbuf, ALLREDUCE_TAG, &tag_param)
                .expect("allreduce latency send");
        }
    }

    // Receive from all peers
    let mut recv_buf = vec![0u8; msg_size];
    for peer in 0..size {
        if peer != rank {
            let req = ctx
                .worker()
                .tag_recv(&mut recv_buf, ALLREDUCE_TAG, TAG_MASK, &tag_param)
                .expect("allreduce latency recv")
                .expect("allreduce latency recv request");
            while !req.check_finished().unwrap_or(false) {
                ctx.progress();
            }
            let peer_offset = peer * msg_size;
            gathered[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
        }
    }

    // Perform local sum (mod 256 for u8)
    for (i, item) in recvbuf.iter_mut().enumerate().take(msg_size) {
        let sum: u16 = (0..size).map(|r| gathered[r * msg_size + i] as u16).sum();
        *item = (sum % 256) as u8;
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
            "Allreduce Latency",
            BenchmarkType::CollectiveLatency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
