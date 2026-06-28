//! CLI argument parsing for OSU Micro-Benchmarks.
//!
//! Matches the C reference flag set from `osu_util_options.h`.

use clap::{Parser, ValueEnum};

/// Minimum message size (bytes). Default: 1
pub const DEFAULT_MIN_MSG_SIZE: usize = 1;
/// Maximum message size (bytes). Default: 1048576 (1 MiB)
pub const DEFAULT_MAX_MSG_SIZE: usize = 1_048_576;
/// Message size increment (multiply by this each step). Default: 2
pub const DEFAULT_MSG_SIZE_INCR: usize = 2;
/// Iterations for small messages. Default: 10000
pub const DEFAULT_ITERATIONS_SMALL: usize = 10_000;
/// Iterations for large messages. Default: 1000
pub const DEFAULT_ITERATIONS_LARGE: usize = 1_000;
/// Skip (warmup) iterations for small messages. Default: 100
pub const DEFAULT_SKIP_SMALL: usize = 100;
/// Skip (warmup) iterations for large messages. Default: 10
pub const DEFAULT_SKIP_LARGE: usize = 10;
/// Cutoff between small and large message sizes. Default: 8192
pub const LARGE_MESSAGE_SIZE: usize = 8192;

/// Accelerator type matching C reference.
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq)]
#[clap(rename_all = "lower")]
pub enum AccelType {
    #[default]
    None,
    Cuda,
    Managed,
    Openacc,
    Rocm,
    Sycl,
}

/// MPI data type for benchmark operations.
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq)]
#[clap(rename_all = "lower")]
pub enum MpiDataType {
    #[default]
    All,
    MpiChar,
    MpiInt,
    MpiFloat,
    MpiDouble,
}

/// Benchmark CLI arguments matching the C reference implementation.
#[derive(Parser, Debug, Clone)]
#[command(
    version = "7.5.2",
    about = "OSU Micro-Benchmarks",
    disable_version_flag = true
)]
pub struct CliArgs {
    /// Minimum message size (bytes), or MIN:MAX:INCR format
    #[arg(short = 'm', long = "message-size", default_value_t = 1)]
    pub min_message_size: usize,

    /// Maximum message size (bytes)
    #[arg(short = 'M', long = "max-message-size", default_value_t = 1_048_576)]
    pub max_message_size: usize,

    /// Number of iterations for timing
    #[arg(short = 'i', long = "iterations", default_value_t = 10_000)]
    pub iterations: usize,

    /// Message size increment (multiplier)
    #[arg(short = 'e', long = "increment", default_value_t = 2)]
    pub message_size_incr: usize,

    /// Number of warmup iterations to skip before timing
    #[arg(short = 'x', long = "skip", default_value_t = 100)]
    pub skip: usize,

    /// Enable DDT (Derived Data Types) support
    #[arg(short = 'd', long = "ddt")]
    pub ddt: bool,

    /// Number of probes
    #[arg(short = 'n', long = "num-probes", default_value_t = 0)]
    pub num_probes: usize,

    /// Print rate (0 or 1)
    #[arg(short = 'r', long = "print-rate", default_value_t = 0)]
    pub print_rate: usize,

    /// Test type (e.g., "all")
    #[arg(long = "test", default_value = "all")]
    pub test: String,

    /// Accelerator type (cuda, managed, openacc, rocm, sycl)
    #[arg(long = "accel", value_enum, default_value_t = AccelType::None)]
    pub accel: AccelType,

    /// Source memory location
    #[arg(long = "src", default_value = "M")]
    pub src: char,

    /// Destination memory location
    #[arg(long = "dst", default_value = "M")]
    pub dst: char,

    /// Enable data validation
    #[arg(long = "validate")]
    pub validate: bool,

    /// Statistics percentile list (e.g., "99,90,50")
    #[arg(long = "stat")]
    pub stat: Option<String>,

    /// Output file path
    #[arg(long = "file")]
    pub file: Option<String>,

    /// MPI data type (all, mpi_char, mpi_int, mpi_float, mpi_double)
    #[arg(long = "dtype", value_enum, default_value_t = MpiDataType::All)]
    pub dtype: MpiDataType,

    /// Root rank for collective operations
    #[arg(long = "root", default_value_t = 0)]
    pub root: usize,

    /// Enable MPI_IN_PLACE support
    #[arg(long = "enable-inplace")]
    pub enable_inplace: bool,

    /// Enable session-based MPI initialization
    #[arg(long = "enable-session")]
    pub enable_session: bool,

    /// Enable tail latency reporting (P99, P90, P50)
    #[arg(long = "enable-tail-latency")]
    pub enable_tail_latency: bool,

    /// Enable graph output
    #[arg(long = "enable-graph")]
    pub enable_graph: bool,

    /// Enable terminal graph output
    #[arg(long = "enable-graph-term")]
    pub enable_graph_term: bool,

    /// Enable PNG graph output
    #[arg(long = "enable-graph-png")]
    pub enable_graph_png: bool,

    /// Enable PDF graph output
    #[arg(long = "enable-graph-pdf")]
    pub enable_graph_pdf: bool,

    /// Enable log validation
    #[arg(long = "enable-log-validation")]
    pub enable_log_validation: bool,

    /// Log validation directory path
    #[arg(long = "enable-log-validation-dir")]
    pub enable_log_validation_dir: Option<String>,

    /// Print full format listing (MIN/MAX latency in addition to AVERAGE)
    #[arg(short = 'f', long = "full")]
    pub full: bool,

    /// Array size for device (GPU) allocation
    #[arg(short = 'a', long = "array-size", default_value_t = 32)]
    pub array_size: usize,

    /// Number of test calls (for non-blocking collectives)
    #[arg(short = 't', long = "num-test-calls", default_value_t = 100)]
    pub num_test_calls: usize,

    /// Window size (number of messages before sync)
    #[arg(short = 'W', long = "window-size", default_value_t = 64)]
    pub window_size: usize,

    /// Number of pairs involved
    #[arg(short = 'p', long = "num-pairs")]
    pub num_pairs: Option<usize>,

    /// Vary the window size
    #[arg(short = 'V', long = "vary-window")]
    pub vary_window: bool,

    /// Buffer number (single or multiple)
    #[arg(short = 'b', long = "buffer-num", default_value = "single")]
    pub buffer_num: String,

    /// Validation warmup iterations
    #[arg(short = 'u', long = "validation-warmup", default_value_t = 5)]
    pub validation_warmup: usize,

    /// Graph output format (tty, png, pdf)
    #[arg(short = 'G', long = "graph")]
    pub graph: Option<String>,

    /// PAPI events and path
    #[arg(short = 'P', long = "papi")]
    pub papi: Option<String>,

    /// DDT type and parameters
    #[arg(short = 'D', long = "derived-data-type")]
    pub ddt_type: Option<String>,

    /// Neighborhood collective configuration
    #[arg(short = 'N', long = "neighborhood")]
    pub neighborhood: Option<String>,

    /// MPI type for data transfer
    #[arg(short = 'T', long = "type")]
    pub mpi_type: Option<String>,

    /// Sync option for one-sided operations
    #[arg(short = 's', long = "sync-option")]
    pub sync_option: Option<String>,

    /// Window option for one-sided operations
    #[arg(short = 'w', long = "win-options")]
    pub win_option: Option<String>,

    /// CUDA target for dummy computation
    #[arg(short = 'R', long = "cuda-target")]
    pub cuda_target: Option<String>,

    /// Number of partitions
    #[arg(short = 'q', long = "partitions", default_value_t = 8)]
    pub num_partitions: usize,

    /// Root rank configuration (fixed:N or rotate)
    #[arg(short = 'k', long = "root-rank", default_value = "fixed:0")]
    pub root_rank: String,

    /// Tail latency percentiles
    #[arg(short = 'z', long = "tail-lat", num_args = 0..=1, default_missing_value = "")]
    pub tail_lat: Option<String>,

    /// Validation flag (with optional log:dir)
    #[arg(short = 'c', long = "validation", num_args = 0..=1, default_missing_value = "")]
    pub validation: Option<String>,

    /// Session-based MPI initialization
    #[arg(short = 'I', long = "session")]
    pub session: bool,

    /// Run with MPI_IN_PLACE
    #[arg(short = 'l', long = "inplace")]
    pub inplace: bool,

    /// Force UCC backend for collective operations (default: auto-detect)
    #[arg(long = "ucc", conflicts_with = "no_ucc")]
    pub ucc: bool,

    /// Disable UCC backend — use UCX tag-matching fallback only
    #[arg(long = "no-ucc", conflicts_with = "ucc")]
    pub no_ucc: bool,
}

impl CliArgs {
    /// Whether UCC backend should be used for collectives.
    /// `true` = use UCC, `false` = UCX fallback only, `None` = auto-detect.
    pub fn ucc_backend(&self) -> Option<bool> {
        if self.ucc {
            Some(true)
        } else if self.no_ucc {
            Some(false)
        } else {
            None // auto-detect
        }
    }

    /// Parse CLI arguments from std::env::args().
    pub fn parse() -> Self {
        Self::parse_from(std::env::args_os())
    }

    /// Get iterations for a given message size (small vs large cutoff).
    pub fn get_iterations(&self, _msg_size: usize) -> usize {
        self.iterations
    }

    /// Get skip count for a given message size.
    pub fn get_skip(&self, _msg_size: usize) -> usize {
        self.skip
    }
}
