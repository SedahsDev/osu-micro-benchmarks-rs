//! OSU MPI_Fetch_and_op Latency Test
//!
//! Measures one-sided Fetch-and-Op atomic latency using UCX AMO primitives.
//! Uses `amo_fadd64` (fetch-and-add with reply) to match MPI_Fetch_and_op(MPI_SUM) semantics.
//! Uses flush for fence-like synchronization.
//!
//! The C reference uses `MPI_Fetch_and_op(sbuf, tbuf, datatype, 1, disp, MPI_SUM, win)`
//! which atomically adds sbuf to the remote value at tbuf and returns the old value in tbuf.
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_fop_latency
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// Run the FOP latency benchmark.
///
/// Rank 0 performs fetch-and-add on Rank 1's registered memory region.
/// Uses flush for fence-like synchronization (matching default FLUSH sync mode).
///
/// The C reference does:
///   MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, MPI_SUM, win);
///   MPI_Win_flush(1, win);
///
/// We use amo_fadd64(operand=sbuf, remote_addr=tbuf) with reply_buffer set.
/// For messages < 8 bytes, we pad to u64. For larger messages, we iterate
/// over 8-byte chunks.
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
                do_fop(ep, &send_buf, remote_addr, rkey);
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
                do_fop(ep, &send_buf, remote_addr, rkey);
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

/// Perform fetch-and-op (fetch-and-add) using amo_fadd64.
///
/// The fetch variants require a reply_buffer to receive the old value.
/// For messages <= 8 bytes, we do a single amo_fadd64.
/// For larger messages, we iterate over 8-byte chunks.
fn do_fop(ep: &ucx_sys::ep::Ep, buf: &[u8], remote_addr: u64, rkey: &ucx_sys::rma::RemoteKey) {
    if buf.is_empty() {
        return;
    }

    if buf.len() <= 8 {
        let mut padded = [0u8; 8];
        padded[..buf.len()].copy_from_slice(buf);
        let operand = u64::from_le_bytes(padded);
        let mut reply: u64 = 0;
        let amo_param = ucx_sys::RequestParamBuilder::new()
            .no_imm_cmpl()
            .reply_buffer(&mut reply as *mut _ as *mut std::os::raw::c_void)
            .build();
        let req = ep
            .amo_fadd64(operand, remote_addr, rkey, &amo_param)
            .expect("amo_fadd64");
        // Wait for the fetch-and-add to complete
        if let Some(r) = req {
            while !r.check_finished().unwrap_or(false) {
                // progress is handled by caller's flush
            }
        }
        let _ = reply; // old value (discarded for benchmark)
    } else {
        let chunks = buf.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let operand = u64::from_le_bytes(chunk.try_into().unwrap());
            let mut reply: u64 = 0;
            let amo_param = ucx_sys::RequestParamBuilder::new()
                .no_imm_cmpl()
                .reply_buffer(&mut reply as *mut _ as *mut std::os::raw::c_void)
                .build();
            let req = ep
                .amo_fadd64(operand, remote_addr, rkey, &amo_param)
                .expect("amo_fadd64");
            if let Some(r) = req {
                while !r.check_finished().unwrap_or(false) {}
            }
            let _ = reply;
        }

        if !remainder.is_empty() {
            let mut padded = [0u8; 8];
            padded[..remainder.len()].copy_from_slice(remainder);
            let operand = u64::from_le_bytes(padded);
            let mut reply: u64 = 0;
            let amo_param = ucx_sys::RequestParamBuilder::new()
                .no_imm_cmpl()
                .reply_buffer(&mut reply as *mut _ as *mut std::os::raw::c_void)
                .build();
            let req = ep
                .amo_fadd64(operand, remote_addr, rkey, &amo_param)
                .expect("amo_fadd64");
            if let Some(r) = req {
                while !r.check_finished().unwrap_or(false) {}
            }
            let _ = reply;
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
        output::print_header(&mut out, "MPI_Fetch_and_op Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
