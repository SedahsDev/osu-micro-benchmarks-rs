# Planned / not yet implemented

## Persistent collectives

All `collective/src/bin/*_persistent.rs` binaries currently **exit 77** with a clear message.
They remain in the workspace for suite layout parity with OSU 7.5.2.

Implementation plan: reuse blocking collective kernels with init/start/wait timing
(MPI-style `MPI_*_init` / `MPI_Startall` / `MPI_Waitall` analogue via UCC or UCX loops).

## Neighborhood

Several `osu_neighbor_*` binaries already use ring topology tag-matching.
See `docs/neighbor_collective_implementation_plan.md` for remaining work.

## message_sizes

Centralized in `osu_common::cli::message_sizes` — do not reintroduce per-binary copies.
