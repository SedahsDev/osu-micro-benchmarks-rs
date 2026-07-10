//! OSU MPI Init/Finalize timing Test (v7.5.2)
//!
//! Measures average time for process init (PMIx/UCX/UCC via `OsUContext::init`)
//! by comparing wall-clock around a single init on this process, then reporting
//! min/avg/max across ranks after a barrier. (Full C OSU may re-init in a loop;
//! we time the one-shot path used by the Rust suite.)
//!
//! ```bash
//! prterun -np 2 ./target/release/osu_init
//! ```

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

fn main() {
    let args = CliArgs::parse();

    // Time includes PMIx_Init + UCX context/worker/endpoints (+ optional UCC).
    let t0 = Wtime::new();
    let ctx = OsUContext::init(args.ucc_backend());
    let init_us = t0.elapsed_us();

    let rank = ctx.rank();
    let size = ctx.size();

    if size < 1 {
        process::exit(1);
    }

    ctx.barrier();

    let min_t = ctx.allreduce_min_f64(init_us);
    let max_t = ctx.allreduce_max_f64(init_us);
    let sum_t = ctx.allreduce_sum_f64(init_us);
    let avg_t = sum_t / size as f64;

    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "Init", BenchmarkType::CollectiveLatency);
        // Reuse latency row: Size unused → 0, times in microseconds.
        output::print_latency_header(&mut out);
        output::print_latency_row(&mut out, 0, avg_t, min_t, max_t);
        output::print_newline(&mut out);
    }

    ctx.barrier();
}
