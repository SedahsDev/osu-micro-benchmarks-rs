//! One-sided RMA benchmarks for the OSU Micro-Benchmarks suite.
//!
//! Implements `osu_put_latency`, `osu_get_latency`, and `osu_acc_latency`
//! using UCX RMA primitives (rma_put, rma_get, amo_add64) with PMIx
//! bootstrap and rkey exchange.
