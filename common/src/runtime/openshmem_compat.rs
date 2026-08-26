//! OpenSHMEM collective compatibility helpers.
//!
//! This module exposes the direct OpenSHMEM collective mappings used by
//! [`super::OsUContext`]. The existing benchmarks also need the UCX endpoint API
//! for point-to-point and unsupported collective variants.

use openshmem::error::Result;
use ucc::collective::UccReductionOp;

/// Initialize the OpenSHMEM runtime used by this compatibility layer.
///
/// OpenSHMEM owns a process-global PMIx/UCX lifecycle. Do not call this while an
/// `OsUContext` is already initialized in the same process.
pub fn init() -> Result<()> {
    openshmem::init::init()
}

/// Finalize the OpenSHMEM runtime. Finalization is idempotent.
pub fn finalize() -> Result<()> {
    openshmem::init::finalize()
}

/// OpenSHMEM-backed barrier.
pub fn barrier() -> Result<()> {
    openshmem::coll::barrier()
}

/// OpenSHMEM-backed byte broadcast. `values` is modified in place on non-root
/// PEs and is the source buffer on the root PE.
pub fn broadcast(root: usize, values: &mut [u8]) -> Result<()> {
    openshmem::coll::broadcast(root, values)
}

/// OpenSHMEM-backed equal-sized byte allgather, in PE rank order.
pub fn collect(values: &[u8]) -> Result<Vec<u8>> {
    openshmem::coll::collect(values)
}

/// OpenSHMEM-backed byte sum allreduce.
pub fn allreduce_sum(values: &mut [u8]) -> Result<()> {
    openshmem::coll::reduce(UccReductionOp::Sum, values)
}

/// Return this PE's rank.
pub fn my_pe() -> Result<u32> {
    openshmem::init::my_pe()
}

/// Return the number of PEs.
pub fn n_pes() -> Result<usize> {
    openshmem::init::n_pes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_surface_compiles() {
        let _: fn() -> Result<()> = init;
        let _: fn() -> Result<()> = finalize;
        let _: fn() -> Result<()> = barrier;
        let _: fn(usize, &mut [u8]) -> Result<()> = broadcast;
        let _: fn(&[u8]) -> Result<Vec<u8>> = collect;
        let _: fn(&mut [u8]) -> Result<()> = allreduce_sum;
    }
}

// Direct UCX implementations in `collective_blocking.rs` remain the fallback
// for APIs not represented by openshmem::coll (gather/scatter/alltoall and
// variable-count and neighborhood variants). Keep those call sites explicitly
// marked TODO(openshmem) rather than silently claiming OpenSHMEM coverage.
