//! OSU MPI Hello World Test (v7.5.2)
//!
//! Each process prints its rank and job size after PMIx/UCX init.
//! Matches the spirit of the C OSU `osu_hello` utility.
//!
//! ```bash
//! prterun -np 4 ./target/release/osu_hello
//! ```

use osu_common::cli::CliArgs;
use osu_common::runtime::OsUContext;
use std::io::{self, Write};

fn main() {
    let args = CliArgs::parse();
    let ctx = OsUContext::init(args.ucc_backend());
    let rank = ctx.rank();
    let size = ctx.size();

    // Synchronize so output is less scrambled (still best-effort with concurrent prints).
    ctx.barrier();

    let mut out = io::stdout().lock();
    if rank == 0 {
        let _ = writeln!(out, "# OSU MPI Hello World Test v7.5.2 (Rust)");
        let _ = writeln!(out, "# Processes: {size}");
    }
    ctx.barrier();

    let _ = writeln!(out, "Hello, World from process {rank} of {size}");
    let _ = out.flush();

    ctx.barrier();
    // ctx drop finalizes runtime
}
