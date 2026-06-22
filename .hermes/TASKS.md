# OSU Micro-Benchmarks Rust Port — Task Tracker

## Task 6: Non-blocking Collectives

Runtime refactored into `common/src/runtime/` module. Non-blocking methods added.
All 20 `osu_i*.rs` binaries still need implementation.

### Pattern for non-blocking benchmark binaries

```rust
//! OSU MPI <Name> Non-Blocking Latency Test (v7.5.2)
//!
//! Requires at least 2 processes.
//!
//! # Usage
//! prterun -np 2 ./target/release/osu_i<name>

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
        if rank == 0 { eprintln!("This test requires at least two processes"); }
        process::exit(1);
    }

    let iterations = args.iterations;
    let skip = args.skip;
    ctx.barrier();

    // For buffer operations, allocate buffers
    // let mut sendbuf = vec![1u8; args.max_message_size];
    // let mut recvbuf = vec![0u8; args.max_message_size];

    let mut timer: f64 = 0.0;
    for i in 0..(iterations + skip) {
        let t_start = Wtime::new();

        // Non-blocking init — returns immediately
        let mut request = ctx.ibarrier();  // or ctx.iallreduce(&sendbuf, &mut recvbuf, msg_size)

        let t_init = Wtime::new();  // init_time measured here if needed

        // Wait for completion
        request.wait();

        let elapsed_us = t_start.elapsed_us();
        if i >= skip { timer += elapsed_us; }
    }

    let latency = timer / iterations as f64;
    let min_time = ctx.allreduce_min_f64(latency);
    let max_time = ctx.allreduce_max_f64(latency);
    let sum_time = ctx.allreduce_sum_f64(latency);
    let avg_time = sum_time / size as f64;

    if rank == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "<Name>", BenchmarkType::NonBlockingCollective);
        output::print_latency_header(&mut out);
        output::print_latency_row(&mut out, 0, avg_time, min_time, max_time);
        output::print_newline(&mut out);
    }
}

fn main() {
    let args = CliArgs::parse();
    let ctx = OsUContext::init();
    if ctx.rank() == 0 {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        output::print_header(&mut out, "<Name>", BenchmarkType::NonBlockingCollective);
        output::print_latency_header(&mut out);
    }
    run_benchmark(&ctx, &args);
}
```

### Non-blocking methods available on OsUContext:
- `ibarrier() -> OsURequest`
- `ibcast(buf: &mut [u8], msg_size: usize, root: usize) -> OsURequest`
- `ireduce(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) -> OsURequest`
- `iallreduce(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) -> OsURequest`
- `iallgather(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) -> OsURequest`
- `igather(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) -> OsURequest`
- `iscatter(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize) -> OsURequest`
- `ialltoall(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize) -> OsURequest`
- `iallgatherv(sendbuf: &[u8], recvbuf: &mut [u8], sendcounts: &[usize], sdispls: &[usize], recvcounts: &[usize], rdispls: &[usize]) -> OsURequest`
- `ialltoallv(sendbuf: &[u8], recvbuf: &mut [u8], sendcounts: &[usize], sdispls: &[usize], recvcounts: &[usize], rdispls: &[usize]) -> OsURequest`
- `ialltoallw(...) -> OsURequest` (same as ialltoallv)
- `ireduce_scatter_block(sendbuf: &[u8], recvbuf: &mut [u8], elemcount: usize) -> OsURequest`
- `ireducescatter(sendbuf: &[u8], recvbuf: &mut [u8], counts: &[usize]) -> OsURequest`
- `igatherv(sendbuf: &[u8], recvbuf: &mut [u8], msg_size: usize, root: usize, recvcounts: &[usize], rdispls: &[usize]) -> OsURequest`
- `iscatterv(sendbuf: &[u8], recvbuf: &mut [u8], root: usize, sendcounts: &[usize], sdispls: &[usize]) -> OsURequest`

### Binaries to implement (20 total):
- [ ] osu_ibarrier — simplest, no buffers
- [ ] osu_iallreduce
- [ ] osu_iallgather
- [ ] osu_ibcast
- [ ] osu_ireduce
- [ ] osu_iscatter
- [ ] osu_igather
- [ ] osu_iallgatherv
- [ ] osu_ialltoall
- [ ] osu_ialltoallv
- [ ] osu_ialltoallw
- [ ] osu_ireduce_scatter
- [ ] osu_ireduce_scatter_block
- [ ] osu_iscatterv
- [ ] osu_igatherv
- [ ] osu_ineighbor_allgather
- [ ] osu_ineighbor_allgatherv
- [ ] osu_ineighbor_alltoall
- [ ] osu_ineighbor_alltoallv
- [ ] osu_ineighbor_alltoallw

### Runtime module layout (after refactor):
```
common/src/runtime/
  mod.rs              — re-exports
  tags.rs             — tag constants
  request.rs          — OsURequest + OsURequestOp + wait()
  context.rs          — OsUContext + blocking methods
  nb_barrier.rs       — ibarrier
  nb_bcast.rs         — ibcast
  nb_reduce.rs        — ireduce, iallreduce
  nb_gather.rs        — iallgather, igather, igatherv
  nb_scatter.rs       — iscatter, iscatterv
  nb_alltoall.rs      — ialltoall, ialltoallv, ialltoallw
  nb_gatherv.rs       — iallgatherv
  nb_reduce_scatter.rs — ireduce_scatter_block, ireducescatter
  helpers.rs          — ep_tag_recv_from
  sys.rs, worker.rs   — UCX bindings (unchanged)
```
