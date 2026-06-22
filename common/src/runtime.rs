//! PMIx/UCX/UCC runtime initialization.
//!
//! Provides functions for initializing and finalizing the underlying
//! communication runtimes. Currently stub implementations — real UCX/PMIx
//! calls will be integrated in subsequent iterations.

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of processes.
    pub num_procs: usize,
    /// Rank of this process.
    pub rank: usize,
    /// UCX transport to use (e.g., "tcp", "shm", "mlx").
    pub transport: String,
    /// Enable CUDA/GPU support.
    pub gpu_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_procs: 2,
            rank: 0,
            transport: "tcp".to_string(),
            gpu_enabled: false,
        }
    }
}

/// Initialize PMIx runtime.
///
/// Returns the rank and size of this process.
pub fn pmix_init() -> (usize, usize) {
    panic!("TODO: PMIx init not yet implemented. Use stub_get_rank/size for now.")
}

/// Finalize PMIx runtime.
pub fn pmix_finalize() {
    panic!("TODO: PMIx finalize not yet implemented.")
}

/// Initialize UCX context and worker.
///
/// Returns a handle to the UCX worker for this process.
pub fn ucx_init(config: &RuntimeConfig) -> UcxHandle {
    let _ = config;
    panic!("TODO: UCX init not yet implemented. Create context and worker here.")
}

/// Finalize UCX resources.
pub fn ucx_finalize(handle: UcxHandle) {
    let _ = handle;
    panic!("TODO: UCX finalize not yet implemented.")
}

/// Create a UCX endpoint to a remote worker.
pub fn ucx_create_endpoint(_handle: &UcxHandle, _remote_address: &[u8]) -> UcxEndpoint {
    panic!("TODO: UCX endpoint creation not yet implemented.")
}

/// Get the packed address of the current worker.
pub fn ucx_pack_address(_handle: &UcxHandle) -> Vec<u8> {
    panic!("TODO: UCX address packing not yet implemented.")
}

/// Progress the UCX worker (poll for completions).
pub fn ucx_progress(_handle: &UcxHandle, _count: u32) -> u32 {
    panic!("TODO: UCX progress not yet implemented.")
}

/// Initialize UCC context.
pub fn ucc_init(_config: &RuntimeConfig) -> UccHandle {
    panic!("TODO: UCC init not yet implemented.")
}

/// Finalize UCC resources.
pub fn ucc_finalize(_handle: UccHandle) {
    panic!("TODO: UCC finalize not yet implemented.")
}

/// Stub handle types — replaced with real UCX/UCC handles later.
#[derive(Debug, Clone, Copy)]
pub struct UcxHandle(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct UcxEndpoint(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct UccHandle(pub usize);

/// Stub implementations for testing without real runtime.
///
/// Use these to verify benchmark logic before integrating real UCX calls.
pub mod stub {
    /// Get the rank from the `OMPI_COMM_WORLD_RANK` or `PMIX_RANK` env var,
    /// or default to 0.
    pub fn get_rank() -> usize {
        std::env::var("OMPI_COMM_WORLD_RANK")
            .or_else(|_| std::env::var("PMIX_RANK"))
            .or_else(|_| std::env::var("MPI_RANK"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Get the size from the `OMPI_COMM_WORLD_SIZE` or `PMIX_SIZE` env var,
    /// or default to 1.
    pub fn get_size() -> usize {
        std::env::var("OMPI_COMM_WORLD_SIZE")
            .or_else(|_| std::env::var("PMIX_SIZE"))
            .or_else(|_| std::env::var("MPI_SIZE"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
    }

    /// Barrier across all processes (stub — no-op for single process).
    pub fn barrier() {
        // In production, this would use PMIx barrier or UCX collective.
    }
}
