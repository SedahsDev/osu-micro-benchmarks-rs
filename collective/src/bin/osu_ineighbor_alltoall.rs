//! OSU MPI Non-blocking Ineighbor_alltoall Latency Test (v7.5.2)
//!
//! Measures non-blocking neighbor alltoall latency using a ring topology.

use osu_common::cli::CliArgs;
use osu_common::output::{self, BenchmarkType};
use osu_common::runtime::OsUContext;
use osu_common::timing::Wtime;
use std::io;
use std::process;

fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This test requires at least two processes");
        }
        process::exit(1);
    }

    let iterations = args.iterations;
    let skip = args.skip;

    ctx.barrier();

    let min_count = 0;
    let max_count = 65536;
    let mut count = min_count;

    loop {
        let msg_size = count;
        let sendbuf = vec![0u8; size * msg_size];
        let mut recvbuf = vec![0u8; size * msg_size];

        let mut timer: f64 = 0.0;
        let mut tcomp_total: f64 = 0.0;
        let mut wait_total: f64 = 0.0;
        let mut init_total: f64 = 0.0;

        for i in 0..(iterations + skip) {
            let t_start = Wtime::new();

            let init_start = Wtime::new();
            let mut request = ctx.ineighbor_alltoall(&sendbuf, &mut recvbuf, msg_size);
            let init_time = init_start.elapsed_us();

            let comp_start = Wtime::new();
            let comp_time = comp_start.elapsed_us();

            let wait_start = Wtime::new();
            request.wait();
            let wait_time = wait_start.elapsed_us();

            let elapsed_us = t_start.elapsed_us();

            if i >= skip {
                timer += elapsed_us;
                tcomp_total += comp_time;
                wait_total += wait_time;
                init_total += init_time;
                ctx.barrier();
            }
        }

        let overlap = timer / iterations as f64;
        let cpu_avg = tcomp_total / iterations as f64;
        let wait_avg = wait_total / iterations as f64;
        let init_avg = init_total / iterations as f64;
        let comm_avg = overlap - init_avg;

        let _overlap_min = ctx.allreduce_min_f64(overlap);
        let _overlap_max = ctx.allreduce_max_f64(overlap);

        if rank == 0 {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            if count == min_count {
                output::print_nbc_header(&mut out);
            }
            output::print_nbc_row(
                &mut out, msg_size, overlap, cpu_avg, comm_avg, wait_avg, init_avg,
            );
            output::print_newline(&mut out);
        }

        if count == 0 {
            count = 1;
        } else if count >= max_count {
            break;
        } else {
            count *= 2;
        }
    }
}

fn main() {
    let args = CliArgs::parse();
    let ctx = OsUContext::init();

    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(
            &mut out,
            "Non-blocking Ineighbor_alltoall",
            BenchmarkType::NonBlockingCollective,
        );
    }

    run_benchmark(&ctx, &args);
}
