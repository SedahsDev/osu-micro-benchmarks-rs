//! OSU MPI_Compare_and_swap Latency Test
//!
//! Measures one-sided Compare-and-Swap (CAS) atomic latency using UCX AMO primitives.
//! Uses `amo_cswap64` (no-fetch compare-and-swap) to match MPI_Compare_and_swap semantics.
//! Uses flush for fence-like synchronization.
//!
//! The C reference uses `MPI_Compare_and_swap(sbuf, cbuf, tbuf, datatype, 1, disp, win)`
//! which compares the remote value at `tbuf` with `cbuf`, and if equal, replaces with `sbuf`.
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_cas_latency
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// Run the CAS latency benchmark.
///
/// Rank 0 performs compare-and-swap on Rank 1's registered memory region.
/// Uses flush for fence-like synchronization (matching default FLUSH sync mode).
///
/// The C reference does:
///   MPI_Compare_and_swap(sbuf, cbuf, tbuf, data_type, 1, disp, win);
///   MPI_Win_flush(1, win);
///
/// We use amo_cswap64(expected=cbuf, replacement=sbuf, remote_addr=tbuf).
/// For messages < 8 bytes, we pad to u64. For messages == 8 bytes, we use the full u64.
/// Larger messages are not meaningful for CAS (single atomic operation), but we
/// iterate over 8-byte chunks to match the C reference's message size loop.
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

    let amo_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();
    let flush_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Allocate buffers — CAS operates on u64 values
        // sbuf = replacement value, cbuf = compare (expected) value
        let mut send_buf = vec![0u8; msg_size];
        let compare_buf = vec![0u8; msg_size];

        // Fill with pattern for debugging
        for (i, byte) in send_buf.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        let remote_addr = ctx.remote_mem_addr(partner);
        let rkey = ctx
            .remote_rkey(partner)
            .expect("RMA context required for one-sided benchmark");

        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            if rank == 0 {
                flush_blocking(worker, &flush_param);
                do_cas(ep, &send_buf, &compare_buf, remote_addr, rkey, &amo_param);
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
                do_cas(ep, &send_buf, &compare_buf, remote_addr, rkey, &amo_param);
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

/// Perform compare-and-swap operation using amo_cswap64.
///
/// For messages <= 8 bytes, we do a single amo_cswap64.
/// For larger messages, we iterate over 8-byte chunks (matching C reference behavior
/// where CAS is called once per datatype element).
fn do_cas(
    ep: &ucx_sys::ep::Ep,
    replacement: &[u8],
    expected: &[u8],
    remote_addr: u64,
    rkey: &ucx_sys::rma::RemoteKey,
    param: &ucx_sys::RequestParam,
) {
    if replacement.is_empty() {
        return;
    }

    if replacement.len() <= 8 {
        let mut rep_bytes = [0u8; 8];
        let mut exp_bytes = [0u8; 8];
        rep_bytes[..replacement.len()].copy_from_slice(replacement);
        exp_bytes[..expected.len()].copy_from_slice(expected);
        let rep_val = u64::from_le_bytes(rep_bytes);
        let exp_val = u64::from_le_bytes(exp_bytes);
        ep.amo_cswap64(exp_val, rep_val, remote_addr, rkey, param)
            .expect("amo_cswap64");
    } else {
        // For larger messages, do CAS per 8-byte chunk
        let rep_chunks = replacement.chunks_exact(8);
        let exp_chunks = expected.chunks_exact(8);
        let rep_rem = rep_chunks.remainder();
        let exp_rem = exp_chunks.remainder();

        for (rep_chunk, exp_chunk) in rep_chunks.zip(exp_chunks) {
            let rep_val = u64::from_le_bytes(rep_chunk.try_into().unwrap());
            let exp_val = u64::from_le_bytes(exp_chunk.try_into().unwrap());
            ep.amo_cswap64(exp_val, rep_val, remote_addr, rkey, param)
                .expect("amo_cswap64");
        }

        // Handle remainder
        if !rep_rem.is_empty() || !exp_rem.is_empty() {
            let mut rep_padded = [0u8; 8];
            let mut exp_padded = [0u8; 8];
            rep_padded[..rep_rem.len()].copy_from_slice(rep_rem);
            exp_padded[..exp_rem.len()].copy_from_slice(exp_rem);
            let rep_val = u64::from_le_bytes(rep_padded);
            let exp_val = u64::from_le_bytes(exp_padded);
            ep.amo_cswap64(exp_val, rep_val, remote_addr, rkey, param)
                .expect("amo_cswap64");
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

/// Generate message sizes from min to max using the given increment.
fn message_sizes(args: &CliArgs) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut size = args.min_message_size;
    while size <= args.max_message_size {
        sizes.push(size);
        if size == 0 {
            size = 1;
        } else {
            if size <= osu_common::cli::LARGE_MESSAGE_SIZE {
                size *= args.message_size_incr;
            } else {
                size += args.message_size_incr;
            }
        }
    }
    sizes
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
        output::print_header(
            &mut out,
            "MPI_Compare_and_swap Latency",
            BenchmarkType::Latency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
