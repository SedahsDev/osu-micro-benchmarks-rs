//! Thin CUDA-runtime FFI provider — the initial [`DeviceProvider`] implementation.
//!
//! # Why the runtime API and not the driver API
//!
//! Benchmarks need to allocate, fill, copy and synchronize. The CUDA *runtime* API is the
//! smallest surface that does all four (`cudaMalloc`, `cudaMemset`, `cudaMemcpy`,
//! `cudaDeviceSynchronize`), and it is available on every NVIDIA install including the DGX
//! Spark's. The driver API would add context/primary-context management for no benefit here.
//!
//! # Scope: this is deliberately a shim
//!
//! This module exists to prove the interface, not to become the permanent binding. It declares
//! exactly the entry points the benchmarks use and nothing else. When the interface has settled,
//! the intent is to replace it with a native binding crate (`cudarc`, or our own `cuda-sys` in
//! the family with `pmix-rs`/`ucx-rs`/`ucc-rs`) by implementing [`DeviceProvider`] over that
//! crate and registering it — no benchmark changes.
//!
//! Symbol resolution is by name at link time; `common/build.rs` locates a CUDA install and emits
//! the search path/link flags + rpath when the `cuda-runtime` feature is on. The feature is off
//! by default so the workspace still builds where no CUDA toolkit exists.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr::NonNull;

use super::MemKind;
use super::provider::{DeviceError, DeviceProvider};

// ── cuda_runtime_api.h, the subset we use ──────────────────────────────────────────────────────
// Types are the C ones: cudaError_t is an enum (int), cudaMemcpyKind an enum (int).

/// `cudaMemcpyHostToDevice`
const MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `cudaMemcpyDeviceToHost`
const MEMCPY_DEVICE_TO_HOST: c_int = 2;
/// `cudaMemAttachGlobal` — accessible from any stream on the device.
const MEM_ATTACH_GLOBAL: c_uint = 1;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
}

/// Translate a `cudaError_t` into a [`DeviceError::Runtime`] carrying the runtime's own string.
fn check(status: c_int, what: &str) -> Result<(), DeviceError> {
    if status == 0 {
        return Ok(());
    }
    let text = unsafe {
        let ptr = cudaGetErrorString(status);
        if ptr.is_null() {
            format!("cuda error {status}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(DeviceError::Runtime(format!(
        "{what}: {text} (cudaError {status})"
    )))
}

/// The CUDA-runtime provider.
pub struct CudaRuntime;

impl CudaRuntime {
    fn device_count_raw(&self) -> Result<usize, DeviceError> {
        let mut count: c_int = 0;
        // A machine with no driver returns an error here, which is "no devices", not a failure
        // worth aborting a benchmark over.
        if unsafe { cudaGetDeviceCount(&mut count) } != 0 {
            return Ok(0);
        }
        Ok(count.max(0) as usize)
    }
}

impl DeviceProvider for CudaRuntime {
    fn name(&self) -> &'static str {
        "cuda-runtime"
    }

    fn is_available(&self) -> bool {
        self.device_count_raw().map(|n| n > 0).unwrap_or(false)
    }

    fn device_count(&self) -> Result<usize, DeviceError> {
        self.device_count_raw()
    }

    fn set_device(&self, ordinal: usize) -> Result<(), DeviceError> {
        let ordinal = c_int::try_from(ordinal)
            .map_err(|_| DeviceError::Usage("device ordinal out of range"))?;
        check(unsafe { cudaSetDevice(ordinal) }, "cudaSetDevice")
    }

    fn alloc(&self, len: usize, kind: MemKind) -> Result<NonNull<u8>, DeviceError> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        match kind {
            MemKind::Device => check(unsafe { cudaMalloc(&mut raw, len) }, "cudaMalloc")?,
            MemKind::Managed => check(
                unsafe { cudaMallocManaged(&mut raw, len, MEM_ATTACH_GLOBAL) },
                "cudaMallocManaged",
            )?,
            MemKind::Host => {
                return Err(DeviceError::Usage(
                    "host memory is not allocated through the device provider",
                ));
            }
        }
        NonNull::new(raw.cast::<u8>()).ok_or_else(|| {
            DeviceError::Runtime(format!(
                "CUDA returned a null pointer for a {len}-byte allocation"
            ))
        })
    }

    unsafe fn free(&self, ptr: NonNull<u8>) {
        // Best-effort: freeing from Drop cannot report, and a failure here (e.g. during
        // teardown after the context is gone) must not panic in a benchmark.
        let _ = check(
            unsafe { cudaFree(ptr.as_ptr().cast::<c_void>()) },
            "cudaFree",
        );
    }

    unsafe fn memset(&self, ptr: NonNull<u8>, value: u8, len: usize) -> Result<(), DeviceError> {
        check(
            unsafe { cudaMemset(ptr.as_ptr().cast::<c_void>(), c_int::from(value), len) },
            "cudaMemset",
        )
    }

    unsafe fn copy_d2h(
        &self,
        dst: *mut u8,
        src: NonNull<u8>,
        len: usize,
    ) -> Result<(), DeviceError> {
        check(
            unsafe {
                cudaMemcpy(
                    dst.cast::<c_void>(),
                    src.as_ptr().cast::<c_void>(),
                    len,
                    MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy(device->host)",
        )
    }

    unsafe fn copy_h2d(
        &self,
        dst: NonNull<u8>,
        src: *const u8,
        len: usize,
    ) -> Result<(), DeviceError> {
        check(
            unsafe {
                cudaMemcpy(
                    dst.as_ptr().cast::<c_void>(),
                    src.cast::<c_void>(),
                    len,
                    MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy(host->device)",
        )
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }
}

/// The provider instance registered as the compile-time default when the feature is on.
pub static CUDA_RUNTIME: CudaRuntime = CudaRuntime;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_count_never_panics_without_a_driver() {
        // Must return a number (possibly 0), not unwind — benchmarks call this for banners.
        let provider = CudaRuntime;
        let count = provider.device_count().expect("device_count");
        eprintln!("cuda-runtime reports {count} device(s)");
    }

    #[test]
    fn host_kind_allocation_is_a_usage_error() {
        let provider = CudaRuntime;
        assert!(matches!(
            provider.alloc(64, MemKind::Host),
            Err(DeviceError::Usage(_))
        ));
    }
}
