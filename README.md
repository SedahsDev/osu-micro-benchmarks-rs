# OSU Micro-Benchmarks in Rust

Rust reimplementation of the [OSU Micro-Benchmarks 7.5.2](https://mvapich.cse.ohio-state.edu/benchmarks/) suite.

Stack: **pmix-rs** (bootstrap) + **ucx-sys** (pt2pt/RMA) + **ucc** (collectives, with UCX fallback).

Full code review: [`REVIEW.md`](./REVIEW.md).

## Build

```bash
export PMIX_PREFIX=/path/to/pmix-or-prrte
export UCX_PREFIX=/path/to/ucx
export UCC_PREFIX=/path/to/ucc
export LD_LIBRARY_PATH=$PMIX_PREFIX/lib:$UCX_PREFIX/lib:$UCC_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cargo build --release --workspace
```

Workspace members: `common`, `pt2pt`, `collective`, `onesided`, `congestion`, `startup`.

See [`../BUILDING.md`](../BUILDING.md).

## Status (truthful snapshot)

Roughly **~50 / 66** binaries have real logic; remaining are stubs (persistent collectives, startup, many neighborhood).

### Point-to-point (`pt2pt/`)

| Benchmark | Status |
|---|---|
| `osu_latency`, `osu_bw`, `osu_bibw` | Implemented |
| `osu_mbw_mr`, `osu_multi_lat` | Implemented |
| `osu_latency_mp`, `osu_latency_mt` | Implemented |
| `osu_partitioned_latency` | Implemented |
| Persistent pt2pt variants | Stub / partial |

### Collectives (`collective/`)

| Class | Status |
|---|---|
| Blocking (allreduce, barrier, bcast, gather*, scatter*, alltoall*, reduce*, …) | Mostly implemented (UCC + UCX fallback) |
| Non-blocking `osu_i*` | Many implemented via `OsURequest` |
| Persistent `*_persistent` | **Stub** (~14 binaries) |
| Neighborhood `osu_neighbor_*` / `osu_ineighbor_*` | **Stub** (plan in `docs/`) |

### One-sided (`onesided/`)

| Class | Status |
|---|---|
| put/get/acc latency & bandwidth, CAS, FOP, etc. | Implemented (RMA; needs RDMA TLS) |

### Congestion

| Benchmark | Status |
|---|---|
| `osu_bw_fan_in`, `osu_bw_fan_out` | Implemented |

### Startup

| Benchmark | Status |
|---|---|
| `osu_hello` | ✅ Implemented |
| `osu_init` | ✅ Implemented |
| Collective `*_persistent` binaries | 🔲 Planned (exit 77) — see `docs/PLANNED.md` |

## Running

Needs a PMIx-aware launcher (`prterun` / `srun`):

```bash
# Point-to-point: 2 processes
prterun -np 2 ./target/release/osu_latency

# Collectives: 2+ processes
prterun -np 4 ./target/release/osu_allreduce

# Onesided RMA: needs RMA-capable UCX TLS (not plain TCP)
prterun -np 2 ./target/release/osu_put_latency
```

CLI options match OSU-style flags where implemented (see each binary `--help`).

## License

BSD-style (see `LICENSE`). Upstream OSU benchmarks have their own license; this is an independent Rust reimplementation.
