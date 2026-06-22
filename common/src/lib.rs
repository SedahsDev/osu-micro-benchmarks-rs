//! Shared utilities for the OSU Micro-Benchmarks Rust reimplementation.
//!
//! Provides CLI argument parsing, output formatting, timing utilities,
//! runtime initialization (PMIx/UCX/UCC), and library path helpers.

pub mod cli;
pub mod libp;
pub mod output;
pub mod runtime;
pub mod timing;
