//! OSU MPI_Accumulate Latency Test
//!
//! Measures one-sided Accumulate latency using UCX AMO (atomic add) primitives.
//! Uses fence-based synchronization (flush) to match MPI_Win_fence semantics.
//!
//! The C reference uses MPI_Accumulate with MPI_INT datatype.
//! We use amo_add64 with u64 operands, matching the count of integers.
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_acc_latency
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// Run the Accumulate latency benchmark.
///
/// Rank 0 performs atomic add (AMO) to Rank 1's registered memory region.
/// Uses flush for fence-like synchronization.
///
/// The C reference does:
///   MPI_Accumulate(sbuf, count, MPI_INT, 1, 0, count, MPI_INT, MPI_SUM, win);
///
/// We use amo_add64 for each u64 word in the message. For small messages
/// (< 8 bytes), we do a single amo_add64. For larger messages, we do
/// multiple amo_add64 calls to cover the full buffer.
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

        // Allocate send buffer with pattern data
        let send_buf = vec![1u8; msg_size];

        let remote_addr = ctx.remote_mem_addr(partner);
        let rkey = ctx
            .remote_rkey(partner)
            .expect("RMA context required for one-sided benchmark");

        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            if rank == 0 {
                flush_blocking(worker, &flush_param);
                do_accumulate(ep, &send_buf, remote_addr, rkey, &rma_param);
                flush_blocking(worker, &flush_param);
            } else {
                flush_blocking(worker, &flush_param);
                flush_blocking(worker, &flush_param);
            }
        }

        // Timed iterations
        for _ in 0..iterations {
            let start = Wtime::new();
            if rank == 0 {
                flush_blocking(worker, &flush_param);
                do_accumulate(ep, &send_buf, remote_addr, rkey, &rma_param);
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

/// Perform accumulate operation using AMO add64.
///
/// For messages >= 8 bytes, we do one amo_add64 per 8-byte chunk.
/// For messages < 8 bytes, we do a single amo_add64 (padding internally).
/// This matches the C reference's MPI_Accumulate with MPI_SUM.
fn do_accumulate(
    ep: &ucx_sys::ep::Ep,
    buf: &[u8],
    remote_addr: u64,
    rkey: &ucx_sys::rma::RemoteKey,
    param: &ucx_sys::RequestParam,
) {
    if buf.is_empty() {
        return;
    }

    // For small messages, use a single amo_add64 with the buffer value
    // padded to 8 bytes
    if buf.len() <= 8 {
        let mut padded = [0u8; 8];
        padded[..buf.len()].copy_from_slice(buf);
        let operand = u64::from_le_bytes(padded);
        ep.amo_add64(operand, remote_addr, rkey, param)
            .expect("amo_add64");
    } else {
        // For larger messages, do amo_add64 for each 8-byte chunk
        let chunks = buf.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let operand = u64::from_le_bytes(chunk.try_into().unwrap());
            ep.amo_add64(operand, remote_addr, rkey, param)
                .expect("amo_add64");
        }

        // Handle remainder bytes
        if !remainder.is_empty() {
            let mut padded = [0u8; 8];
            padded[..remainder.len()].copy_from_slice(remainder);
            let operand = u64::from_le_bytes(padded);
            ep.amo_add64(operand, remote_addr, rkey, param)
                .expect("amo_add64");
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
    let max_size = args.max_message_size;
    let mut rma_target = vec![0u8; max_size];

    // Create the unified runtime context with RMA support
    let ctx = OsUContext::init_with_rma(args.ucc_backend(), Some(&mut rma_target));

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "MPI_Accumulate Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
