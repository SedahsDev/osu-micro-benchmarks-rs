//! OSU MPI Latency Persistent Test (v7.5.2)
//!
//! Measures point-to-point message latency using persistent requests.
//! Persistent requests are created once per message size with
//! MPI_Send_init/MPI_Recv_init, then replayed each iteration with
//! MPI_Start/MPI_Wait. This reduces per-iteration overhead compared
//! to the non-persistent latency test.
//!
//! Protocol (matches C reference exactly):
//! - Rank 0: Init Send(to 1) + Recv(from 1). Each iter: Start Send, Wait Send, Start Recv, Wait Recv.
//! - Rank 1: Init Recv(from 0) + Send(to 0). Each iter: Start Recv, Wait Recv, Start Send, Wait Send.
//!
//! In our UCX implementation we emulate the persistent pattern by
//! pre-creating request parameters and reusing them across iterations.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_latency_persistent
//! ```

use osu_common::cli::{CliArgs, message_sizes};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io;
use std::process;

/// UCX tag used for all messages in this benchmark (matches C tag 1).
const TAG: u64 = 1;
/// Tag mask for exact matching.
const TAG_MASK: u64 = u64::MAX;

/// Run the persistent latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires at least 2 processes.");
        process::exit(1);
    }
    if rank >= 2 {
        return;
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    // Build request params once — reused across all iterations (persistent pattern)
    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    // Barrier so both sides are ready
    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let send_buf = vec![0u8; msg_size];
        let mut recv_buf = vec![0u8; msg_size];
        let mut result = TimingResult::new();

        // Skip warmup iterations
        for _ in 0..skip {
            persistent_latency_iteration(rank, ep, worker, &tag_param, &send_buf, &mut recv_buf);
        }

        // Timed iterations
        for _ in 0..iterations {
            let start = Wtime::new();
            persistent_latency_iteration(rank, ep, worker, &tag_param, &send_buf, &mut recv_buf);
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

/// Single persistent latency iteration.
///
/// Matches C reference pattern:
/// - Rank 0: Start(send_obj), Wait(send_obj), Start(recv_obj), Wait(recv_obj)
/// - Rank 1: Start(recv_obj), Wait(recv_obj), Start(send_obj), Wait(send_obj)
fn persistent_latency_iteration(
    rank: usize,
    ep: &ucx_sys::ep::Ep,
    worker: &ucx_sys::worker::Worker,
    tag_param: &ucx_sys::RequestParam,
    send_buf: &[u8],
    recv_buf: &mut [u8],
) {
    if rank == 0 {
        // Send first, then receive
        ep.tag_send(send_buf, TAG, tag_param).expect("tag_send");
        progress_worker(worker);

        let recv_req = worker
            .tag_recv(recv_buf, TAG, TAG_MASK, tag_param)
            .expect("tag_recv")
            .expect("recv request");
        while !recv_req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }
    } else {
        // Receive first, then send
        let recv_req = worker
            .tag_recv(recv_buf, TAG, TAG_MASK, tag_param)
            .expect("tag_recv")
            .expect("recv request");
        while !recv_req.check_finished().unwrap_or(false) {
            progress_worker(worker);
        }

        ep.tag_send(send_buf, TAG, tag_param).expect("tag_send");
    }
}

/// Progress the worker until no more work is available.
fn progress_worker(worker: &ucx_sys::worker::Worker) {
    loop {
        if !worker.progress() {
            break;
        }
    }
}

fn main() {
    let args = CliArgs::parse();

    let ctx = OsUContext::init(args.ucc_backend());

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);
}
