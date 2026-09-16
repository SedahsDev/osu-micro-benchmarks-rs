# GPU communication benchmarks

Device-memory support for the OSU micro-benchmarks. This document covers what exists on this
branch, how to build it, what is deliberately still missing, and what the target hardware implies.

## Status

| Piece | State |
|---|---|
| Device memory abstraction (`osu_common::device`) | **implemented**, unit-tested on the GPU |
| Swappable provider seam (`DeviceProvider`) | **implemented** |
| Thin CUDA-runtime provider (`cuda-runtime` feature) | **implemented**, verified on Beast (RTX A2000 + RTX 3090 Ti, both `sm_86`) |
| UCX pointer-based transfer entry points | **implemented** on `ucx-rs` branch `feat/device-memory-support` |
| Benchmark wiring (`--accel cuda\|managed` actually used by a benchmark) | **not yet** — next step |
| One-sided (put/get) device variants | planned (Phase 2) |
| GPU collectives | planned (Phase 3) — see "Collectives" below |

## Why the device layer is behind a provider seam

Benchmarks touch only `DeviceBuffer` and `MemKind`. Everything backend-specific is behind
`DeviceProvider`, and the active provider is resolved at runtime:

```rust
// compile-time default (this branch): thin CUDA-runtime FFI shim when the feature is on
cargo build --features cuda-runtime

// runtime override — the seam that makes the shim replaceable:
osu_common::device::set_provider(&MyNativeProvider);
```

The thin FFI shim exists to settle the *interface*, not to become the permanent binding. Replacing
it with a native binding crate (`cudarc`, or a `cuda-sys` in the same family as
`pmix-rs`/`ucx-rs`/`ucc-rs`) means implementing `DeviceProvider` in a new module and registering
it. **No benchmark changes, no call-site changes.**

The trait is deliberately narrow — allocate, fill, copy, synchronize, device ordinal. Streams,
kernels and graphs do not belong here; these are bandwidth and latency benchmarks, not a compute
framework.

## Building

The `cuda-runtime` feature is **off by default** so the workspace still builds on hosts with no
CUDA toolkit. With it off, any device request returns `DeviceError::Unsupported` and `--accel
cuda` fails cleanly rather than silently doing something wrong.

```bash
# CPU-only / build-host default
cargo build

# accelerator host
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 cargo build --features cuda-runtime
cargo test -p osu-common --features cuda-runtime
```

`common/build.rs` locates the toolkit in this order: `CUDA_TOOLKIT_PATH` → `CUDA_HOME` →
`CUDA_PATH` → versioned `/usr/local/cuda-*` (newest first) → `/usr/local/cuda` → `/opt/cuda`. It
emits the link search path, `-lcudart`, and an RPATH. If the feature is on and no toolkit is
found, the build stops with the environment variable to set rather than a bare linker error.

Verified on Beast (CUDA 13.3, RTX A2000 + RTX 3090 Ti, both `sm_86`):

```
cuda-runtime reports 2 device(s)
test device::tests::device_round_trip_when_available ... ok
```

That test allocates device memory, copies a 1 KiB pattern host→device, synchronizes, copies back
and asserts byte equality, then repeats with `cudaMemset` and with managed memory. It is skipped
(not failed) on hosts with no accelerator, so the suite stays green in CPU-only CI.

## The UCX side (and why it lives in ucx-rs)

UCX transfers in `ucx-rs` all took `&[u8]` / `&mut [u8]`. **A device pointer must never be
materialised as a host slice** — the host cannot dereference it — so a slice-based API cannot
express device memory at all. The companion branch adds pointer+length variants:

| Added | Where | Note |
|---|---|---|
| `Ep::tag_send_ptr` | `ucx-rs` | unsafe; pairs with `memory_type()` |
| `Worker::tag_recv_ptr` | `ucx-rs` | unsafe; receive is a **worker** op in UCX, not an ep op |
| `Ep::rma_put_ptr` / `Ep::rma_get_ptr` | `ucx-rs` | unsafe |

It also re-exports `ucs_memory_type_t` / `ucs_thread_mode_t`, which were **private-to-the-crate
despite appearing in public signatures** — downstream code could not name the type it was required
to pass. That is the same defect currently failing `openshmem-rs` CI with
`error[E0603]: enum ucs_thread_mode_t is private`.

Every device transfer must declare its memory type:

```rust
let params = ucx_sys::RequestParamBuilder::new()
    .memory_type(buffer.ucx_memory_type())   // UCS_MEMORY_TYPE_CUDA | _CUDA_MANAGED
    .build();
```

Declaring it is not just an optimisation. Without it UCX probes the pointer, and a UCX built
without CUDA support will classify a device pointer as host memory and take a path that cannot
work.

## Remaining work to wire a benchmark

Pilot: `osu_latency`, then `osu_bw`, `osu_bibw`, `osu_mbw_mr`.

1. Resolve `--accel` → `MemKind` via `MemKind::from_accel(args.accel)` (already rejects
   openacc/rocm/sycl explicitly).
2. Replace the per-size `vec![0u8; msg_size]` with either a `Vec<u8>` (host) or a `DeviceBuffer`,
   branching once per size, not per iteration.
3. Branch the transfer call: slice API for host, `*_ptr` API + `memory_type()` for device.
4. Print the memory kind in the output line so a result can never be misread as a host number.

## Target hardware notes (DGX Spark / GB10)

- **Unified memory changes what is worth measuring.** Grace and Blackwell share 128 GB LPDDR5x, so
  H2D/D2H copies are effectively no-ops. Use `--accel managed` (`UCS_MEMORY_TYPE_CUDA_MANAGED`);
  treating Spark like a discrete-GPU host and reporting copy bandwidth would measure nothing real.
- **No GPU Direct RDMA** (`nvidia-peermem` does not load). Inter-node device traffic stages
  through host memory, capping real bandwidth well below the NIC's line rate. Report the staging
  path explicitly — that penalty is the most interesting number this suite can produce on Spark.
- **Requires a CUDA-enabled UCX.** Verify with `ucx_info -d | grep -i cuda` (expect `cuda_copy`,
  `cuda_ipc`). A UCX built without `--with-cuda` has zero CUDA symbols and cannot move device
  memory regardless of what this crate does.
- **Do not assume device-memory atomics work over RoCE.** UCX AMO against CUDA memory is
  unverified on `sm_121`; test before designing around it.

## Collectives (Phase 3)

On Spark, GPU collectives are a **NCCL** story — NCCL ships with the platform stack and is
RoCE-aware — so the upstream `osu_nccl_*` family is the model to follow.

Rust bindings for NCCL exist but are all alpha with negligible adoption
(`baracuda-nccl` 0.0.1-alpha.x, ~1.8k downloads; `singe-nccl` 0.1.0-alpha.7, ~130). Note the
crates.io name collision: the crate literally called `nccl` is an unrelated config-format
library.

Two viable routes, to be decided with the hardware in hand:

1. **UCC with its NCCL transport layer.** `ucc-rs` already exposes `UCC_MEMORY_TYPE_CUDA` and
   `UCC_EE_CUDA_STREAM`, and NVIDIA's UCC can be built `--with-nccl` so `tl_nccl` services the
   collectives. This reuses the binding family we already have. Caveat: the local UCC 1.9.0 is
   built `--with-ucx` only — no CUDA, no NCCL — so this needs a rebuild either way.
2. **`nccl-rs` / `nccl-sys` in our own family**, matching how `pmix-rs` / `ucx-rs` / `ucc-rs` are
   done. `nccl.h` is a single header and the API surface is small; this gives exact `nccl-tests`
   parity and no alpha dependency.

Recommendation: try (1) first since the binding exists; write (2) only if we need NCCL-specific
semantics that UCC's transport layer does not surface.

## Verification gates

Nothing here can be validated on Beast beyond the allocation/copy layer: its UCX 1.20.1 has no
CUDA support (`ucx_info -d` lists no cuda transports, `libucm.so` has zero CUDA symbols) and it
has no RDMA transport at all. The gate for both this and the `gups-rs` work is a Spark with a
CUDA-enabled UCX.
