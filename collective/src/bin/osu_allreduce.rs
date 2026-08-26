//! OSU MPI Allreduce Latency Test (v7.5.2)
//!
//! Measures allreduce latency using UCC allreduce (with UCX fallback).
//!
//! Requires at least 2 processes. Runs allreduce with MPI_SUM for each
//! message size from min to max, reporting min/avg/max latency per size.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_allreduce
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;
use ucc::collective::{DataType, ReductionOp};

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
    let mut sendbuf = vec![1u8; max_buf_size];
    let mut recvbuf = vec![0u8; max_buf_size];

    // Barrier so all processes are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Use the slice for this message size
        let send_slice = &mut sendbuf[..msg_size];
        let recv_slice = &mut recvbuf[..msg_size];

        // Warmup iterations
        for _ in 0..skip {
            allreduce_blocking(ctx, send_slice, recv_slice, msg_size);
            ctx.barrier();
        }

        let mut timer: f64 = 0.0;

        // Timed iterations
        for _ in 0..iterations {
            let t_start = Wtime::new();
            allreduce_blocking(ctx, send_slice, recv_slice, msg_size);
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

/// Perform a blocking allreduce using UCC if available, otherwise UCX fallback.
fn allreduce_blocking(ctx: &OsUContext, sendbuf: &mut [u8], recvbuf: &mut [u8], msg_size: usize) {
    if ctx.openshmem_allreduce(sendbuf, recvbuf) {
        return;
    }
    if let Some(team) = ctx.ucc_team() {
        // Use UCC allreduce with CHAR datatype
        let req = match team.allreduce(sendbuf, DataType::Uchar, ReductionOp::Sum) {
            Ok(req) => req,
            Err(_) => {
                // UCC failed, fall back to UCX
                allreduce_ucx_fallback(ctx, sendbuf, recvbuf, msg_size);
                return;
            }
        };
        while !req.test().unwrap_or(false) {
            ctx.progress();
        }
        // UCC allreduce writes to the same buffer (in-place), copy to recvbuf
        recvbuf.copy_from_slice(sendbuf);
    } else {
        allreduce_ucx_fallback(ctx, sendbuf, recvbuf, msg_size);
    }
}

/// UCX-based allreduce fallback using all-gather + local sum.
fn allreduce_ucx_fallback(ctx: &OsUContext, sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size <= 1 {
        recvbuf.copy_from_slice(sendbuf);
        return;
    }

    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
    const ALLREDUCE_TAG: u64 = 0xABCDEF00;
    const TAG_MASK: u64 = u64::MAX;

    // Gather all data
    let total_size = msg_size * size;
    let mut gathered = vec![0u8; total_size];

    // Place our own data
    let my_offset = rank * msg_size;
    gathered[my_offset..my_offset + msg_size].copy_from_slice(sendbuf);

    // Send our data to all peers
    for peer in 0..size {
        if peer != rank {
            ctx.endpoint(peer)
                .tag_send(sendbuf, ALLREDUCE_TAG, &tag_param)
                .expect("allreduce send");
        }
    }

    // Receive from all peers
    let mut recv_buf = vec![0u8; msg_size];
    for peer in 0..size {
        if peer != rank {
            let req = ctx
                .worker()
                .tag_recv(&mut recv_buf, ALLREDUCE_TAG, TAG_MASK, &tag_param)
                .expect("allreduce recv")
                .expect("allreduce recv request");
            while !req.check_finished().unwrap_or(false) {
                ctx.progress();
            }
            let peer_offset = peer * msg_size;
            gathered[peer_offset..peer_offset + msg_size].copy_from_slice(&recv_buf);
        }
    }

    // Perform local sum
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
        output::print_header(&mut out, "Allreduce", BenchmarkType::CollectiveLatency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
