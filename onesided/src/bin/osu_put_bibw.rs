//! OSU MPI_Put Bi-directional Bandwidth Test
//!
//! Measures bidirectional one-sided Put bandwidth using UCX RMA primitives.
//! Both ranks simultaneously perform RMA Puts to each other's registered memory
//! regions, using flush for fence-like synchronization.
//!
//! Matches C reference behavior (PSCW/fence sync):
//!   Both ranks do:
//!     MPI_Win_fence(0, win)
//!     for j in 0..window_size:
//!         MPI_Put(sbuf + j*size, size, MPI_CHAR, partner, disp + j*size, size, MPI_CHAR, win)
//!     MPI_Win_fence(0, win)
//!
//! Bandwidth is multiplied by 2 since both directions contribute.
//!
//! Requires exactly 2 processes.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_put_bibw
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

/// Run the Put bidirectional bandwidth benchmark.
///
/// Both ranks perform RMA Puts to each other's registered memory regions.
/// Uses flush for fence-like synchronization (matching MPI_Win_fence).
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

    let window_size = args.window_size;

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        // Allocate send buffer large enough for window_size * msg_size
        let send_buf = vec![0u8; msg_size * window_size];

        let remote_addr = ctx.remote_mem_addr(partner);
        let rkey = ctx
            .remote_rkey(partner)
            .expect("RMA context required for one-sided benchmark");

        // Skip warmup iterations
        for _ in 0..skip {
            flush_blocking(worker, &flush_param);
            do_put_bw(
                ep,
                &send_buf,
                msg_size,
                window_size,
                remote_addr,
                rkey,
                &rma_param,
            );
            flush_blocking(worker, &flush_param);
        }

        // Timed iterations — both ranks participate
        let mut total_us: f64 = 0.0;
        for _ in 0..iterations {
            let start = Wtime::new();
            flush_blocking(worker, &flush_param);
            do_put_bw(
                ep,
                &send_buf,
                msg_size,
                window_size,
                remote_addr,
                rkey,
                &rma_param,
            );
            flush_blocking(worker, &flush_param);
            total_us += start.elapsed_us();
        }

        if rank == 0 {
            // Bidirectional: multiply by 2 since both directions contribute
            let total_bytes = msg_size as f64 * window_size as f64 * iterations as f64;
            let total_time_s = total_us / 1_000_000.0;
            let bandwidth_mbps = total_bytes / total_time_s / 1_000_000.0 * 2.0;

            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_bandwidth_avg(&mut out, msg_size, bandwidth_mbps);
            output::print_newline(&mut out);
        }
    }
}

/// Perform window_size Put operations from send_buf.
fn do_put_bw(
    ep: &ucx_sys::ep::Ep,
    send_buf: &[u8],
    msg_size: usize,
    window_size: usize,
    remote_addr: u64,
    rkey: &ucx_sys::rma::RemoteKey,
    param: &ucx_sys::RequestParam,
) {
    for j in 0..window_size {
        let offset = j * msg_size;
        let remote_offset = (remote_addr as i64 + offset as i64) as u64;
        let slice = &send_buf[offset..offset + msg_size];
        ep.rma_put(slice, remote_offset, rkey, param)
            .expect("rma_put");
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
    // Buffer needs to be large enough for max_message_size * window_size
    let max_size = args.max_message_size * args.window_size;
    let mut rma_target = vec![0u8; max_size];

    // Create the unified runtime context with RMA support
    let ctx = OsUContext::init_with_rma(args.ucc_backend(), Some(&mut rma_target));

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "MPI_Put Bi-directional Bandwidth",
            BenchmarkType::Bandwidth,
        );
        output::print_bandwidth_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
