//! PMIx/UCX/UCC runtime initialization.
//!
//! Provides a unified `OsUContext` that wraps communication layers:
//! - **PMIx** — bootstrap, rank/size discovery, address exchange
//! - **UCX** — point-to-point tag matching, endpoints, progress engine
//! - **UCC** — native collective operations built on top of UCX
//!
//! UCC is initialized with OOB (out-of-band) callbacks that use UCX tag
//! matching for the allgather needed during context/team creation.
//!
//! Pattern derived from gups-rs which uses the same stack for the GUPS benchmark.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let ctx = OsUContext::init().expect("init");
//! let partner = 1 - ctx.rank();
//! // ... use ctx.endpoint(partner), ctx.worker(), ctx.ucc_team(), etc.
//! drop(ctx); // finalizes UCC → UCX → PMIx
//! ```

mod collective_blocking;
mod constants;
mod context;
mod helpers;
mod non_blocking;
mod ucc_oob;

pub mod openshmem_compat;

pub use context::OsUContext;
pub use non_blocking::OsURequest;
