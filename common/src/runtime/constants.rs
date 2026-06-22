//! Constants and imports shared across the runtime module.

// ── PMIx key names for data exchange ──

pub const PMIX_KEY_UCX_ADDR: &str = "osu.ucx.addr";

// ── UCC OOB tag ──
// Tag base for UCC out-of-band allgather. Each peer uses 0xCC0000 + peer_rank.
pub const UCC_OOB_TAG_BASE: u64 = 0xCC0000;
pub const UCC_OOB_TAG_MASK: u64 = u64::MAX;
