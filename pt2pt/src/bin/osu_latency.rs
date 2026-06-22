//! OSU MPI Latency Test (v7.5.2)
//!
//! Measures point-to-point message latency using UCX tag matching.
//!
//! Requires exactly 2 processes. Runs ping-pong Send/Recv for each
//! message size from min to max, reporting min/avg/max latency per size.
//!
//! # Usage
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_latency
//! ```

use osu_common::cli::{CliArgs, LARGE_MESSAGE_SIZE};
use osu_common::output::{self, BenchmarkType};
use osu_common::timing::{TimingResult, Wtime};
use std::io::{self, Write};
use std::process;

/// UCX tag used for all messages in this benchmark.
const TAG: u64 = 0x123456789ABCDEF0;
/// Tag mask for exact matching.
const TAG_MASK: u64 = u64::MAX;

/// Run the latency benchmark using UCX tag matching.
fn run_ucx_benchmark(args: &CliArgs) {
    let rank = osu_common::runtime::stub::get_rank();
    let size = osu_common::runtime::stub::get_size();

    if size < 2 {
        eprintln!("Error: This test requires at least 2 processes.");
        process::exit(1);
    }
    if rank >= 2 {
        // Only ranks 0 and 1 participate.
        return;
    }

    let partner = 1 - rank;

    // TODO: Initialize UCX context and worker
    // let config = osu_common::runtime::RuntimeConfig {
    //     num_procs: size,
    //     rank,
    //     transport: "tcp".to_string(),
    //     gpu_enabled: false,
    // };
    // let handle = osu_common::runtime::ucx_init(&config);

    // TODO: Exchange addresses and create endpoints
    // let my_addr = osu_common::runtime::ucx_pack_address(&handle);
    // let remote_addr = if rank == 0 {
    //     // Receive address from rank 1
    //     let mut buf = vec![0u8; 4096];
    //     osu_common::runtime::ucx_tag_recv(&handle, &mut buf, TAG, TAG_MASK);
    //     buf
    // } else {
    //     // Send address to rank 0
    //     osu_common::runtime::ucx_tag_send(&handle, &my_addr, TAG);
    //     my_addr
    // };
    // let ep = osu_common::runtime::ucx_create_endpoint(&handle, &remote_addr);

    // TODO: Run ping-pong latency test
    // for msg_size in message_sizes(args) {
    //     let iterations = args.get_iterations(msg_size);
    //     let skip = args.get_skip(msg_size);
    //
    //     let mut buf = vec![0u8; msg_size];
    //     let mut result = TimingResult::new();
    //
    //     // Skip warmup iterations
    //     for _ in 0..skip {
    //         if rank == 0 {
    //             osu_common::runtime::ucx_tag_send(&ep, &buf, TAG);
    //             osu_common::runtime::ucx_tag_recv(&handle, &mut buf, TAG, TAG_MASK);
    //         } else {
    //             osu_common::runtime::ucx_tag_recv(&handle, &mut buf, TAG, TAG_MASK);
    //             osu_common::runtime::ucx_tag_send(&ep, &buf, TAG);
    //         }
    //     }
    //
    //     // Timed iterations
    //     for _ in 0..iterations {
    //         let start = Wtime::new();
    //         if rank == 0 {
    //             osu_common::runtime::ucx_tag_send(&ep, &buf, TAG);
    //             osu_common::runtime::ucx_tag_recv(&handle, &mut buf, TAG, TAG_MASK);
    //         } else {
    //             osu_common::runtime::ucx_tag_recv(&handle, &mut buf, TAG, TAG_MASK);
    //             osu_common::runtime::ucx_tag_send(&ep, &buf, TAG);
    //         }
    //         let elapsed = start.elapsed_us();
    //         result.add(elapsed);
    //     }
    //
    //     if rank == 0 {
    //         output::print_latency_row(&mut io::stdout(), msg_size, result.avg_us, result.min_us, result.max_us);
    //         output::print_newline(&mut io::stdout());
    //     }
    // }

    // TODO: Finalize UCX
    // osu_common::runtime::ucx_finalize(handle);
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
            // Use multiplication for large messages, addition for small
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

    // Print header (only rank 0)
    let rank = osu_common::runtime::stub::get_rank();
    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Latency", BenchmarkType::Latency);
        output::print_latency_header(&mut out);
    }

    run_ucx_benchmark(&args);
}
