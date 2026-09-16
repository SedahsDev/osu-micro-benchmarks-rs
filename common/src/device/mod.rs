//! Device (accelerator) memory support for the OSU benchmarks.
//!
//! # Why this exists
//!
//! Every benchmark in this workspace was written against host `Vec<u8>` buffers. Accelerator
//! benchmarks move the *same* messages through device memory instead, which changes three things:
//!
//! 1. **Allocation** — device memory is not host memory; it cannot be a `Vec<u8>`.
//! 2. **UCX memory type** — UCX must be told the buffer is device or managed memory
//!    (`RequestParamBuilder::memory_type`), otherwise it probes the pointer and can pick a
//!    host-only transport.
//! 3. **Transfer API** — a device pointer can never be turned into a host `&[u8]`, so the
//!    pointer-based UCX entry points (`Ep::tag_send_ptr`, `Worker::tag_recv_ptr`,
//!    `Ep::rma_put_ptr`, `Ep::rma_get_ptr`) are required.
//!
//! # Swappability (read this before adding a provider)
//!
//! Benchmarks only ever touch [`DeviceBuffer`] and [`MemKind`]. All provider-specific code lives
//! behind the [`DeviceProvider`] trait, and the active provider is resolved at runtime through
//! [`provider`] / [`set_provider`]. That is deliberate: the initial provider is a thin
//! CUDA-runtime FFI shim (`cuda_runtime` module, `cuda-runtime` feature), which we expect to be
//! replaced by a native binding crate (e.g. `cudarc`) once the interface is proven.
//!
//! Adding a provider therefore costs **one new module plus one registration**, and zero changes
//! to any benchmark:
//!
//! ```ignore
//! // in a new src/device/cudarc_provider.rs
//! impl DeviceProvider for CudarcProvider { /* ... */ }
//!
//! // registration — either compile-time default or runtime override:
//! device::set_provider(&CudarcProvider);
//! ```
//!
//! # Feature gating
//!
//! The CUDA-runtime provider is behind the `cuda-runtime` feature (off by default) so that this
//! crate still builds on hosts with no CUDA toolkit. With the feature off, every device request
//! returns [`DeviceError::Unsupported`] and `--accel cuda` fails cleanly instead of misbehaving.

mod kinds;
mod provider;

#[cfg(feature = "cuda-runtime")]
mod cuda_runtime;

pub use kinds::MemKind;
pub use provider::{DeviceError, DeviceProvider};

use std::ptr::NonNull;
use std::sync::OnceLock;

/// Registry slot holding the active provider.
///
/// A `OnceLock` rather than a mutable static: providers are `Sync`, benchmarks are often
/// multi-threaded, and reading the provider must not need a lock on the hot path.
static PROVIDER: OnceLock<&'static dyn DeviceProvider> = OnceLock::new();

/// Provider used when the `cuda-runtime` feature is off or nothing has been registered.
///
/// Compiled out when a provider is available at compile time, so it cannot become a silent
/// fallback that masks a missing CUDA build.
#[cfg(not(feature = "cuda-runtime"))]
struct NullProvider;

#[cfg(not(feature = "cuda-runtime"))]
impl DeviceProvider for NullProvider {
    fn name(&self) -> &'static str {
        "none"
    }
    fn is_available(&self) -> bool {
        false
    }
    fn device_count(&self) -> Result<usize, DeviceError> {
        Ok(0)
    }
    fn set_device(&self, _ordinal: usize) -> Result<(), DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
    fn alloc(&self, _len: usize, _kind: MemKind) -> Result<NonNull<u8>, DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
    unsafe fn free(&self, _ptr: NonNull<u8>) {}
    unsafe fn memset(&self, _ptr: NonNull<u8>, _value: u8, _len: usize) -> Result<(), DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
    unsafe fn copy_d2h(
        &self,
        _dst: *mut u8,
        _src: NonNull<u8>,
        _len: usize,
    ) -> Result<(), DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
    unsafe fn copy_h2d(
        &self,
        _dst: NonNull<u8>,
        _src: *const u8,
        _len: usize,
    ) -> Result<(), DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
    fn synchronize(&self) -> Result<(), DeviceError> {
        Err(DeviceError::Unsupported("no device provider registered"))
    }
}

/// Register the process-wide device provider. Idempotent: the first registration wins.
///
/// This is the seam that lets a native binding crate replace the thin FFI shim without touching
/// a single benchmark. Returns the provider actually in effect (`false` means one was already
/// registered, and the argument was ignored).
pub fn set_provider(candidate: &'static dyn DeviceProvider) -> bool {
    PROVIDER.set(candidate).is_ok()
}

/// The active device provider — never panics, never `None`.
///
/// Resolution order: an explicitly registered provider ([`set_provider`]) wins; otherwise the
/// compile-time default for this build is used — the CUDA-runtime shim when the `cuda-runtime`
/// feature is on, or [`NullProvider`] (every device call returns `Unsupported`) when it is off.
pub fn provider() -> &'static dyn DeviceProvider {
    if let Some(registered) = PROVIDER.get() {
        return *registered;
    }
    #[cfg(feature = "cuda-runtime")]
    {
        &cuda_runtime::CUDA_RUNTIME
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        &NullProvider
    }
}

/// True when a provider is registered *and* it can see at least one device.
pub fn available() -> bool {
    let p = provider();
    p.is_available() && p.device_count().map(|n| n > 0).unwrap_or(false)
}

/// Human-readable description of the active provider, for benchmark banners.
pub fn provider_name() -> &'static str {
    provider().name()
}

/// A buffer that lives in device (or managed) memory.
///
/// Behaves like the `Vec<u8>` it replaces for the purpose of a benchmark loop: sized allocation,
/// fill, host round-trip, and a raw pointer for UCX. Dropping it frees the device allocation.
///
/// # Thread safety
///
/// `Send` but not `Sync`: the allocation is owned by exactly one thread at a time, matching how
/// the benchmarks stage buffers per rank. Sharing a device buffer across threads needs explicit
/// synchronization in CUDA too, so we do not pretend otherwise.
pub struct DeviceBuffer {
    ptr: NonNull<u8>,
    len: usize,
    kind: MemKind,
}

// The pointer is an owned device allocation; moving it between threads is safe, aliasing it is not.
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `len` bytes of [`MemKind`] memory.
    pub fn alloc(kind: MemKind, len: usize) -> Result<Self, DeviceError> {
        if len == 0 {
            return Err(DeviceError::Usage("zero-length device allocation"));
        }
        if kind == MemKind::Host {
            return Err(DeviceError::Usage(
                "DeviceBuffer requires device or managed memory; use Vec<u8> for host buffers",
            ));
        }
        let ptr = provider().alloc(len, kind)?;
        Ok(Self { ptr, len, kind })
    }

    /// Raw pointer for UCX (`tag_send_ptr`, `rma_put_ptr`, …).
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer for UCX receive paths.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Length in bytes, as passed to UCX.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Which memory kind this buffer was allocated as — feeds `memory_type()` on UCX requests.
    pub fn kind(&self) -> MemKind {
        self.kind
    }

    /// The UCX memory type for this buffer, for `RequestParamBuilder::memory_type`.
    pub fn ucx_memory_type(&self) -> ucx_sys::ucs_memory_type_t {
        self.kind.ucx_memory_type()
    }

    /// Fill the whole buffer with `value` (device-side, asynchronous; call [`Self::synchronize`]).
    pub fn fill(&mut self, value: u8) -> Result<(), DeviceError> {
        unsafe { provider().memset(self.ptr, value, self.len) }
    }

    /// Copy host bytes into the buffer.
    pub fn copy_from_slice(&mut self, src: &[u8]) -> Result<(), DeviceError> {
        if src.len() != self.len {
            return Err(DeviceError::Usage(
                "copy_from_slice length does not match device buffer size",
            ));
        }
        unsafe { provider().copy_h2d(self.ptr, src.as_ptr(), self.len) }
    }

    /// Copy the buffer back into host bytes.
    pub fn copy_to_slice(&self, dst: &mut [u8]) -> Result<(), DeviceError> {
        if dst.len() != self.len {
            return Err(DeviceError::Usage(
                "copy_to_slice length does not match device buffer size",
            ));
        }
        unsafe { provider().copy_d2h(dst.as_mut_ptr(), self.ptr, self.len) }
    }

    /// Convenience round-trip for tests and verification paths.
    pub fn to_vec(&self) -> Result<Vec<u8>, DeviceError> {
        let mut out = vec![0u8; self.len];
        self.copy_to_slice(&mut out)?;
        Ok(out)
    }

    /// Wait for outstanding work on the current device to finish.
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        provider().synchronize()
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { provider().free(self.ptr) }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &format_args!("{:p}", self.ptr.as_ptr()))
            .field("len", &self.len)
            .field("kind", &self.kind)
            .field("provider", &provider_name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_kind_is_rejected_for_device_buffers() {
        // Catches a benchmark passing --accel none and accidentally taking the device path.
        assert!(matches!(
            DeviceBuffer::alloc(MemKind::Host, 64),
            Err(DeviceError::Usage(_))
        ));
    }

    #[test]
    fn zero_length_allocation_is_rejected() {
        assert!(matches!(
            DeviceBuffer::alloc(MemKind::Device, 0),
            Err(DeviceError::Usage(_))
        ));
    }

    #[test]
    fn mem_kind_maps_to_ucx_memory_type() {
        assert_eq!(
            MemKind::Device.ucx_memory_type(),
            ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_CUDA
        );
        assert_eq!(
            MemKind::Managed.ucx_memory_type(),
            ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_CUDA_MANAGED
        );
        assert_eq!(
            MemKind::Host.ucx_memory_type(),
            ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_HOST
        );
    }

    #[test]
    fn mismatched_lengths_are_rejected_before_touching_the_device() {
        let mut buf = match DeviceBuffer::alloc(MemKind::Device, 64) {
            Ok(b) => b,
            Err(_) => return, // no provider/GPU on this host — covered by the gpu tests
        };
        assert!(matches!(
            buf.copy_from_slice(&[0u8; 32]),
            Err(DeviceError::Usage(_))
        ));
        let mut short = [0u8; 32];
        assert!(matches!(
            buf.copy_to_slice(&mut short),
            Err(DeviceError::Usage(_))
        ));
    }

    /// Requires a registered provider and at least one visible GPU.
    /// Skipped (not failed) when the host has no accelerator, so the suite stays green on
    /// CPU-only machines — the GPU behaviour is exercised on the DGX Spark.
    #[test]
    fn device_round_trip_when_available() {
        if !available() {
            eprintln!(
                "skipping device_round_trip_when_available: provider={} available=false",
                provider_name()
            );
            return;
        }
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        let mut buf = DeviceBuffer::alloc(MemKind::Device, payload.len()).expect("device alloc");
        buf.copy_from_slice(&payload).expect("h2d copy");
        buf.synchronize().expect("sync");
        let back = buf.to_vec().expect("d2h copy");
        assert_eq!(back, payload, "device memory round-trip must be byte-exact");

        buf.fill(0xAB).expect("memset");
        buf.synchronize().expect("sync");
        assert!(buf.to_vec().expect("d2h").iter().all(|&b| b == 0xAB));

        // Managed memory should behave identically for the host-visible round trip.
        if provider().is_available() {
            let mut managed =
                DeviceBuffer::alloc(MemKind::Managed, payload.len()).expect("managed alloc");
            managed.copy_from_slice(&payload).expect("h2d copy");
            managed.synchronize().expect("sync");
            assert_eq!(managed.to_vec().expect("d2h copy"), payload);
            assert_eq!(
                managed.ucx_memory_type(),
                ucx_sys::ucs_memory_type_t::UCS_MEMORY_TYPE_CUDA_MANAGED
            );
        }
    }
}
