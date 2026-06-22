# OSU Micro-Benchmarks in Rust

Rust reimplementation of the [OSU Micro-Benchmarks 7.5.2](https://mvapich.cse.ohio-state.edu/benchmarks/) test suite.

## Tech Stack

- **pmix-rs** — PMIx bindings for process management and bootstrap
- **ucx-sys** — UCX (Unified Communication X) bindings for point-to-point and RMA communication
- **ucc** — UCC (Unified Collective Communication) bindings for collective operations

## Build

```bash
cargo build --release
```

## Benchmarks

### Point-to-Point (`pt2pt/`)

| Benchmark | Description | Status |
|---|---|---|
| `osu_latency` | Message latency | ✅ Implemented |
| `osu_bw` | Message bandwidth | 🔲 Stub |
| `osu_bibw` | Bidirectional bandwidth | 🔲 Stub |
| `osu_latency_mp` | Multi-process latency | 🔲 Stub |
| `osu_latency_mt` | Multi-threaded latency | 🔲 Stub |
| `osu_mbw_mr` | Multi-buffer multi-region bandwidth | 🔲 Stub |
| `osu_multi_lat` | Multi-latency | 🔲 Stub |
| `osu_partitioned_latency` | Partitioned latency | 🔲 Stub |

### Collectives (`collective/`)

#### Blocking
| Benchmark | Description | Status |
|---|---|---|
| `osu_allreduce` | Allreduce | 🔲 Stub |
| `osu_allgather` | Allgather | 🔲 Stub |
| `osu_allgatherv` | Allgatherv | 🔲 Stub |
| `osu_alltoall` | Alltoall | 🔲 Stub |
| `osu_alltoallv` | Alltoallv | 🔲 Stub |
| `osu_alltoallw` | Alltoallw | 🔲 Stub |
| `osu_barrier` | Barrier | 🔲 Stub |
| `osu_bcast` | Broadcast | 🔲 Stub |
| `osu_gather` | Gather | 🔲 Stub |
| `osu_gatherv` | Gatherv | 🔲 Stub |
| `osu_reduce` | Reduce | 🔲 Stub |
| `osu_reduce_scatter` | Reduce-scatter | 🔲 Stub |
| `osu_reduce_scatter_block` | Reduce-scatter (blocking) | 🔲 Stub |
| `osu_scatter` | Scatter | 🔲 Stub |
| `osu_scatterv` | Scatterv | 🔲 Stub |

#### Non-blocking
| Benchmark | Description | Status |
|---|---|---|
| `osu_iallreduce` | Iallreduce | 🔲 Stub |
| `osu_iallgather` | Iallgather | 🔲 Stub |
| `osu_iallgatherv` | Iallgatherv | 🔲 Stub |
| `osu_ialltoall` | Ialltoall | 🔲 Stub |
| `osu_ialltoallv` | Ialltoallv | 🔲 Stub |
| `osu_ialltoallw` | Ialltoallw | 🔲 Stub |
| `osu_ibarrier` | Ibarrier | 🔲 Stub |
| `osu_ibcast` | Ibcast | 🔲 Stub |
| `osu_igather` | Igather | 🔲 Stub |
| `osu_igatherv` | Igatherv | 🔲 Stub |
| `osu_ireduce` | Ireduce | 🔲 Stub |
| `osu_ireduce_scatter` | Ireduce-scatter | 🔲 Stub |
| `osu_ireduce_scatter_block` | Ireduce-scatter (blocking) | 🔲 Stub |
| `osu_iscatter` | Iscatter | 🔲 Stub |
| `osu_iscatterv` | Iscatterv | 🔲 Stub |

#### Persistent
| Benchmark | Description | Status |
|---|---|---|
| `osu_allreduce_persistent` | Persistent allreduce | 🔲 Stub |
| `osu_allgather_persistent` | Persistent allgather | 🔲 Stub |
| `osu_allgatherv_persistent` | Persistent allgatherv | 🔲 Stub |
| `osu_alltoall_persistent` | Persistent alltoall | 🔲 Stub |
| `osu_alltoallv_persistent` | Persistent alltoallv | 🔲 Stub |
| `osu_alltoallw_persistent` | Persistent alltoallw | 🔲 Stub |
| `osu_barrier_persistent` | Persistent barrier | 🔲 Stub |
| `osu_bcast_persistent` | Persistent bcast | 🔲 Stub |
| `osu_gather_persistent` | Persistent gather | 🔲 Stub |
| `osu_gatherv_persistent` | Persistent gatherv | 🔲 Stub |
| `osu_reduce_persistent` | Persistent reduce | 🔲 Stub |
| `osu_reduce_scatter_persistent` | Persistent reduce-scatter | 🔲 Stub |
| `osu_scatter_persistent` | Persistent scatter | 🔲 Stub |
| `osu_scatterv_persistent` | Persistent scatterv | 🔲 Stub |

#### Neighborhood
| Benchmark | Description | Status |
|---|---|---|
| `osu_neighbor_allgather` | Neighbor allgather | 🔲 Stub |
| `osu_neighbor_allgatherv` | Neighbor allgatherv | 🔲 Stub |
| `osu_neighbor_alltoall` | Neighbor alltoall | 🔲 Stub |
| `osu_neighbor_alltoallv` | Neighbor alltoallv | 🔲 Stub |
| `osu_neighbor_alltoallw` | Neighbor alltoallw | 🔲 Stub |
| `osu_ineighbor_allgather` | Ineighbor allgather | 🔲 Stub |
| `osu_ineighbor_allgatherv` | Ineighbor allgatherv | 🔲 Stub |
| `osu_ineighbor_alltoall` | Ineighbor alltoall | 🔲 Stub |
| `osu_ineighbor_alltoallv` | Ineighbor alltoallv | 🔲 Stub |
| `osu_ineighbor_alltoallw` | Ineighbor alltoallw | 🔲 Stub |

### Congestion (`congestion/`)

| Benchmark | Description | Status |
|---|---|---|
| `osu_bw_fan_in` | Bandwidth fan-in | 🔲 Stub |
| `osu_bw_fan_out` | Bandwidth fan-out | 🔲 Stub |

### Startup (`startup/`)

| Benchmark | Description | Status |
|---|---|---|
| `osu_hello` | Hello world startup | 🔲 Stub |
| `osu_init` | Init/finalize startup | 🔲 Stub |

## Running

Benchmarks require a PMIx-aware launcher (e.g., `prterun` or `srun`):

```bash
# Point-to-point benchmarks require exactly 2 processes
prterun -np 2 ./target/release/osu_latency

# Collective benchmarks require 2+ processes
prterun -np 4 ./target/release/osu_allreduce

# Startup benchmarks can run with any number of processes
prterun -np 2 ./target/release/osu_init
```

## License

MIT
