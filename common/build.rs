use std::env;
use std::path::{Path, PathBuf};

/// Discover UCC include/lib dirs (same logic as ucc-rs build.rs).
/// Order: UCC_PREFIX → UCC_INCLUDE_DIR/UCC_LIB_DIR → $HOME/.local/ucc → common prefixes → /usr
fn discover_ucc() -> (PathBuf, PathBuf) {
    println!("cargo:rerun-if-env-changed=UCC_PREFIX");
    println!("cargo:rerun-if-env-changed=UCC_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=UCC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=HOME");

    if let Ok(prefix) = env::var("UCC_PREFIX") {
        let prefix = PathBuf::from(prefix);
        return (prefix.join("include"), prefix.join("lib"));
    }

    let include = env::var("UCC_INCLUDE_DIR").ok().map(PathBuf::from);
    let lib = env::var("UCC_LIB_DIR").ok().map(PathBuf::from);
    if let (Some(inc), Some(lib)) = (include, lib) {
        return (inc, lib);
    }

    // Check $HOME/.local/ucc (common for `./configure --prefix=$HOME/.local`)
    if let Ok(home) = env::var("HOME") {
        let home_prefix = PathBuf::from(&home).join(".local").join("ucc");
        let inc = home_prefix.join("include");
        let lib = home_prefix.join("lib");
        if inc.join("ucc").join("api").join("ucc.h").exists()
            || inc.join("ucc.h").exists()
            || lib.join("libucc.so").exists()
            || lib.join("libucc.so.1").exists()
        {
            return (inc, lib);
        }
    }

    let fixed_candidates = ["/usr", "/usr/local", "/opt/ucc"];
    for c in fixed_candidates {
        let p = Path::new(c);
        let inc = p.join("include");
        let lib = p.join("lib");
        if inc.join("ucc").join("api").join("ucc.h").exists()
            || inc.join("ucc.h").exists()
            || lib.join("libucc.so").exists()
            || lib.join("libucc.so.1").exists()
        {
            return (inc, lib);
        }
    }

    (PathBuf::from("/usr/include"), PathBuf::from("/usr/lib"))
}

/// Discover a CUDA toolkit install.
///
/// Order: `CUDA_TOOLKIT_PATH` → `CUDA_HOME` → `CUDA_PATH` → versioned dirs (newest first) →
/// `/usr/local/cuda` → `/opt/cuda`. Mirrors the search order that cuTile Rust documents, so a
/// host set up for either toolkit flavour is found by both.
fn discover_cuda() -> (PathBuf, PathBuf) {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let mut candidates: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Ok(prefix) = env::var(var)
            && !prefix.is_empty()
        {
            candidates.push(PathBuf::from(prefix));
        }
    }
    // Newest toolkit first: a host may carry several, and the benchmarks want the one its driver
    // was built against rather than whichever happened to sort first.
    let mut versioned: Vec<PathBuf> = std::fs::read_dir("/usr/local")
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("cuda-"))
                })
                .collect()
        })
        .unwrap_or_default();
    versioned.sort_by(|a, b| b.cmp(a));
    candidates.extend(versioned);
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/opt/cuda"));

    for prefix in candidates {
        let include = prefix.join("include");
        let lib = prefix.join("lib64");
        let lib_fallback = prefix.join("lib");
        let lib = if lib.join("libcudart.so").exists() {
            lib
        } else {
            lib_fallback
        };
        if include.join("cuda_runtime.h").exists() || lib.join("libcudart.so").exists() {
            return (include, lib);
        }
    }

    // Nothing found: fail with an actionable message rather than emitting a bare link error.
    panic!(
        "the `cuda-runtime` feature is enabled but no CUDA toolkit was found.\n\
         Set CUDA_TOOLKIT_PATH (or CUDA_HOME/CUDA_PATH) to the toolkit prefix, e.g.\n\
         \x20 CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 cargo build --features cuda-runtime\n\
         Or build without accelerator support: cargo build"
    );
}

fn main() {
    let (_include_dir, lib_dir) = discover_ucc();

    // Propagate UCC native lib search path and rpath to downstream binaries
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=ucc");
    // Use RPATH (not RUNPATH) so the library is found at runtime without LD_LIBRARY_PATH
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Accelerator support is opt-in: the workspace must still build where no CUDA toolkit exists.
    if env::var_os("CARGO_FEATURE_CUDA_RUNTIME").is_some() {
        let (cuda_include, cuda_lib) = discover_cuda();
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib.display());
        println!(
            "cargo:warning=cuda-runtime enabled: linking libcudart from {} (headers: {})",
            cuda_lib.display(),
            cuda_include.display()
        );
    }
}
