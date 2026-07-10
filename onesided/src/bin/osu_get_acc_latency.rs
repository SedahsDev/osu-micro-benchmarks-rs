//! OSU MPI_Get_accumulate Latency Test
//!
//! Measures one-sided Get_accumulate latency using UCX AMO + RMA primitives.
//! Uses amo_fadd64 (fetch-and-add) to match MPI_Get_accumulate(MPI_SUM) semantics.
//! Uses flush for fence-like synchronization.
//!
//! The C reference uses:
//!   MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size, MPI_CHAR, MPI_SUM, win);
//! which atomically adds sbuf to the remote location and stores the old value in cbuf.
//!
//! Unlike CAS/FOP, get_accumulate operates on variable-length byte buffers (MPI_CHAR).
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_get_acc_latency
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// Run the Get_accumulate latency benchmark.
///
/// Rank 0 performs fetch-and-add (get_accumulate) on Rank 1's registered memory region.
/// Uses flush for fence-like synchronization.
///
/// The C reference does:
///   MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size, MPI_CHAR, MPI_SUM, win);
///   MPI_Win_flush(1, win);
///
/// We emulate this with amo_fadd64 per 8-byte chunk of the buffer.
/// The reply buffer receives the old value (matching the cbuf in MPI_Get_accumulate).
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

        // Allocate send buffer (sbuf) and compare/receive buffer (cbuf)
        let send_buf = vec![1u8; msg_size];
        let mut compare_buf = vec![0u8; msg_size];

        let remote_addr = ctx.remote_mem_addr(partner);
        let rkey = ctx
            .remote_rkey(partner)
            .expect("RMA context required for one-sided benchmark");

        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            if rank == 0 {
                flush_blocking(worker, &flush_param);
                do_get_acc(ep, &send_buf, &mut compare_buf, remote_addr, rkey);
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
                do_get_acc(ep, &send_buf, &mut compare_buf, remote_addr, rkey);
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

/// Perform get_accumulate using amo_fadd64 per 8-byte chunk.
///
/// The reply buffer receives the old value at each chunk, matching
/// MPI_Get_accumulate's behavior of storing original data in cbuf.
fn do_get_acc(
    ep: &ucx_sys::ep::Ep,
    send_buf: &[u8],
    compare_buf: &mut [u8],
    remote_addr: u64,
    rkey: &ucx_sys::rma::RemoteKey,
) {
    if send_buf.is_empty() {
        return;
    }

    if send_buf.len() <= 8 {
        let mut padded = [0u8; 8];
        padded[..send_buf.len()].copy_from_slice(send_buf);
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
        // Store old value in compare buffer
        let reply_bytes = reply.to_le_bytes();
        compare_buf[..send_buf.len()].copy_from_slice(&reply_bytes[..send_buf.len()]);
    } else {
        let chunk_count = send_buf.len() / 8;
        let remainder_offset = chunk_count * 8;

        for i in 0..chunk_count {
            let chunk_offset = i * 8;
            let operand =
                u64::from_le_bytes(send_buf[chunk_offset..chunk_offset + 8].try_into().unwrap());
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
            compare_buf[chunk_offset..chunk_offset + 8].copy_from_slice(&reply.to_le_bytes());
        }

        if remainder_offset < send_buf.len() {
            let remainder = &send_buf[remainder_offset..];
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
            let reply_bytes = reply.to_le_bytes();
            let comp_rem = &mut compare_buf[remainder_offset..];
            if !comp_rem.is_empty() {
                let copy_len = remainder.len().min(comp_rem.len());
                comp_rem[..copy_len].copy_from_slice(&reply_bytes[..copy_len]);
            }
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
        output::print_header(
            &mut out,
            "MPI_Get_accumulate Latency",
            BenchmarkType::Latency,
        );
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
