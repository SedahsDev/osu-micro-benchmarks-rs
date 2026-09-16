//! The provider seam: everything provider-specific hides behind [`DeviceProvider`].

use std::ptr::NonNull;

use super::MemKind;

/// Errors from the device layer.
///
/// Deliberately small and `Display`: benchmark binaries report these next to the message size
/// that failed, so the text is what matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// The requested accelerator/memory kind is not implemented by this build.
    Unsupported(&'static str),
    /// The caller asked for something contradictory (bad lengths, wrong kind for a device buffer).
    Usage(&'static str),
    /// No device provider is registered, or it reports no usable devices.
    NoDevice,
    /// Driver/runtime failure, carrying whatever the backend could tell us.
    Runtime(String),
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::Unsupported(what) => write!(f, "unsupported: {what}"),
            DeviceError::Usage(msg) => write!(f, "usage error: {msg}"),
            DeviceError::NoDevice => write!(f, "no usable accelerator device"),
            DeviceError::Runtime(msg) => write!(f, "device runtime error: {msg}"),
        }
    }
}

impl std::error::Error for DeviceError {}

/// A source of accelerator memory.
///
/// Implement this to swap in a different backend (native binding crate, ROCm, a test double).
/// The trait is intentionally narrow: allocation, fill, copy, synchronize, and the device
/// ordinal. Anything richer (streams, kernels) does not belong here — benchmarks need buffers,
/// not a compute framework.
///
/// # Contract
///
/// - `alloc` returns memory valid for `len` bytes; the caller frees exactly once via `free`.
/// - `free` must tolerate the pointer produced by this provider's own `alloc` only.
/// - `copy_*` must be synchronous enough that a subsequent `synchronize` is not required for
///   correctness of the *host* side of a `copy_d2h`; see each provider's notes.
/// - Methods that take `NonNull<u8>` are `unsafe` because a wrong pointer is UB — the provider
///   cannot validate it.
pub trait DeviceProvider: Send + Sync {
    /// Short name for banners, e.g. `"cuda-runtime"`.
    fn name(&self) -> &'static str;

    /// Cheap, non-allocating check that the backend is usable at all.
    fn is_available(&self) -> bool;

    /// Number of visible devices (0 is valid and means "none here").
    fn device_count(&self) -> Result<usize, DeviceError>;

    /// Select the active device.
    fn set_device(&self, ordinal: usize) -> Result<(), DeviceError>;

    /// Allocate `len` bytes of the given kind. `MemKind::Host` is a caller error.
    fn alloc(&self, len: usize, kind: MemKind) -> Result<NonNull<u8>, DeviceError>;

    /// Free a pointer previously returned by this provider's [`DeviceProvider::alloc`].
    ///
    /// # Safety
    /// `ptr` must come from this provider and must not be used afterwards.
    unsafe fn free(&self, ptr: NonNull<u8>);

    /// Fill `len` bytes at `ptr` with `value`.
    ///
    /// # Safety
    /// `ptr` must be valid for writes of `len` bytes.
    unsafe fn memset(&self, ptr: NonNull<u8>, value: u8, len: usize) -> Result<(), DeviceError>;

    /// Device → host copy.
    ///
    /// # Safety
    /// `dst` must be valid for writes of `len` bytes; `src` must be a device allocation of at
    /// least `len` bytes from this provider.
    unsafe fn copy_d2h(
        &self,
        dst: *mut u8,
        src: NonNull<u8>,
        len: usize,
    ) -> Result<(), DeviceError>;

    /// Host → device copy.
    ///
    /// # Safety
    /// `dst` must be a device allocation of at least `len` bytes from this provider; `src` must
    /// be valid for reads of `len` bytes.
    unsafe fn copy_h2d(
        &self,
        dst: NonNull<u8>,
        src: *const u8,
        len: usize,
    ) -> Result<(), DeviceError>;

    /// Block until outstanding work on the current device completes.
    fn synchronize(&self) -> Result<(), DeviceError>;
}
