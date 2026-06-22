//! OSU MPI Multi-threaded Latency Test (v7.5.2)
//!
//! Measures point-to-point message latency using UCX tag matching.
//! The multi-threaded variant uses Rust std::thread for concurrent
//! coordination while the main thread handles all UCX operations
//! (since UCX workers are not thread-safe).
//!
//! Protocol (matches C reference):
//! - Rank 0: sender side — sends then receives for each iteration
//! - Rank 1: receiver side — receives then sends for each iteration
//! - Worker threads are spawned to coordinate timing and iteration
//!   management, but UCX calls happen in the main thread
//! - Threads use channels to communicate iteration counts
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_latency_mt
//! ```

use osu_common::cli::{CliArgs, LARGE_MESSAGE_SIZE};
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::{TimingResult, Wtime};
use std::io::{self, Write};
use std::process;

/// UCX tag used for all messages in this benchmark.
const TAG: u64 = 0x123456789ABCDEF0;
/// Tag mask for exact matching.
const TAG_MASK: u64 = u64::MAX;

/// Number of threads for the multi-threaded benchmark.
const NUM_THREADS: usize = 1;

/// Run the multi-threaded latency benchmark.
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        eprintln!("Error: This test requires exactly two processes.");
        process::exit(1);
    }
    if rank >= 2 {
        return;
    }

    let num_threads = NUM_THREADS;

    // Print thread info (only rank 0)
    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        let _ = writeln!(out, "# Number of Sender threads: {}", num_threads);
        let _ = writeln!(out, "# Number of Receiver threads: {}", num_threads);
        let _ = out.flush();
    }

    let partner = 1 - rank;
    let ep = ctx.endpoint(partner);
    let worker = ctx.worker();

    let tag_param = ucx_sys::RequestParamBuilder::new().no_imm_cmpl().build();

    ctx.barrier();

    for msg_size in message_sizes(args) {
        let iterations = args.get_iterations(msg_size);
        let skip = args.get_skip(msg_size);

        let mut buf = vec![0u8; msg_size];
        let mut result = TimingResult::new();

        // Use channels to coordinate between main thread and worker threads
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (iter_tx, iter_rx) = std::sync::mpsc::channel();

        // Spawn worker threads that just count iterations
        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let done_tx_clone = done_tx.clone();
            let iter_tx_clone = iter_tx.clone();

            let handle = std::thread::spawn(move || {
                let total = iterations + skip;
                let mut count = 0;
                for i in (thread_id..total).step_by(num_threads) {
                    if i >= skip {
                        count += 1;
                    }
                    let _ = iter_tx_clone.send(i);
                }
                let _ = done_tx_clone.send(count);
            });

            handles.push(handle);
        }

        // Drop the original senders so channels close when threads finish
        drop(done_tx);
        drop(iter_tx);

        // Main thread performs the actual UCX ping-pong operations
        let total = iterations + skip;
        for i in 0..total {
            let timed = i >= skip;
            let start = if timed { Some(Wtime::new()) } else { None };

            if rank == 0 {
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
                ctx.progress();
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
            } else {
                let recv_req = worker
                    .tag_recv(&mut buf, TAG, TAG_MASK, &tag_param)
                    .expect("tag_recv")
                    .expect("recv request");
                while !recv_req.check_finished().unwrap_or(false) {
                    ctx.progress();
                }
                ep.tag_send(&buf, TAG, &tag_param).expect("tag_send");
            }

            if let Some(s) = start {
                let elapsed = s.elapsed_us();
                result.add(elapsed);
            }

            // Drain iteration channel to keep threads in sync
            let _ = iter_rx.try_recv();
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("thread join failed");
        }

        // Drain done channel
        let mut total_timed = 0;
        for count in done_rx {
            total_timed += count;
        }
        let _ = total_timed;

        // Print results (only rank 0)
        if rank == 0 {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            output::print_latency_avg(&mut out, msg_size, result.avg_us);
            output::print_newline(&mut out);
        }

        // Barrier between message sizes
        ctx.barrier();
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
            // Use multiplication for small messages, addition for large
            if size <= LARGE_MESSAGE_SIZE {
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

    // Create the unified runtime context
    let ctx = OsUContext::init();

    // Print header (only rank 0)
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Multi-threaded Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_benchmark(&ctx, &args);

    // ctx dropped here — finalizes UCC → UCX → PMIx in order
}
