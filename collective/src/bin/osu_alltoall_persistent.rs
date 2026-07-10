//! OSU MPI osu_alltoall_persistent (v7.5.2) — **not implemented**
//!
//! Persistent collective variants are planned (Phase D+). This binary exists so
//! the workspace member list matches the OSU suite layout.
//!
//! Status: planned / not implemented. Exit code 77 (skip).
//!
//! See `README.md` and `docs/` for neighborhood/persistent plans.

fn main() {
    eprintln!(
        "# osu_alltoall_persistent: not implemented (persistent collectives planned)\n         # Exit 77 — skipped"
    );
    std::process::exit(77);
}
