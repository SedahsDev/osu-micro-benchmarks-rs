//! Memory kinds a benchmark can ask for, and their UCX memory-type mapping.

use crate::cli::AccelType;

use super::DeviceError;

/// Where a benchmark buffer lives.
///
/// Mirrors the accelerator subset of the C reference's `--accel` values, narrowed to the ones
/// this workspace supports. OpenACC/ROCm/SYCL are deliberately absent rather than silently
/// mapped onto CUDA — [`MemKind::from_accel`] rejects them with a clear error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemKind {
    /// Ordinary host memory — `Vec<u8>`, the pre-existing path.
    Host,
    /// Device memory (`cudaMalloc`). Not host-addressable.
    Device,
    /// Unified/managed memory (`cudaMallocManaged`). Host-addressable, migrates on access.
    ///
    /// The natural default on hardware with unified memory (e.g. DGX Spark / GB10), where a
    /// separate host/device copy is not a real copy at all.
    Managed,
}

impl MemKind {
    /// Map a `--accel` value onto a memory kind.
    pub fn from_accel(accel: AccelType) -> Result<Self, DeviceError> {
        match accel {
            AccelType::None => Ok(MemKind::Host),
            AccelType::Cuda => Ok(MemKind::Device),
            AccelType::Managed => Ok(MemKind::Managed),
            AccelType::Openacc => Err(DeviceError::Unsupported("--accel openacc")),
            AccelType::Rocm => Err(DeviceError::Unsupported("--accel rocm")),
            AccelType::Sycl => Err(DeviceError::Unsupported("--accel sycl")),
        }
    }

    /// True when this kind needs the device provider at all.
    pub fn is_device(self) -> bool {
        !matches!(self, MemKind::Host)
    }

    /// The UCX memory type to pass as `UCP_OP_ATTR_FIELD_MEMORY_TYPE` on send/recv/put/get.
    ///
    /// Declaring the type matters even for correctness of transport selection: without it UCX
    /// probes the buffer, and on a build without CUDA support it will classify a device pointer
    /// as host memory and quietly take a path that cannot work.
    pub fn ucx_memory_type(self) -> ucx_sys::ucs_memory_type_t {
        match self {
            MemKind::Host => ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_HOST,
            MemKind::Device => ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_CUDA,
            MemKind::Managed => ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_CUDA_MANAGED,
        }
    }

    /// Label used in benchmark output so a result is never ambiguous about which memory moved.
    pub fn label(self) -> &'static str {
        match self {
            MemKind::Host => "host",
            MemKind::Device => "device",
            MemKind::Managed => "managed",
        }
    }
}

impl std::fmt::Display for MemKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
