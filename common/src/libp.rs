//! Library path helpers for finding `.so` files at runtime.
//!
//! Manipulates `LD_LIBRARY_PATH` to ensure shared libraries can be found
//! by the dynamic linker. Useful when UCX/PMIx libraries are installed
//! in non-standard locations.

use std::env;

/// Add a directory to `LD_LIBRARY_PATH` if not already present.
///
/// Returns `true` if the path was added, `false` if it was already there.
pub fn add_to_ld_library_path(path: &str) -> bool {
    let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if current.contains(path) {
        return false;
    }

    let new_path = if current.is_empty() {
        path.to_string()
    } else {
        format!("{}:{}", current, path)
    };

    unsafe {
        env::set_var("LD_LIBRARY_PATH", &new_path);
    }
    true
}

/// Remove a directory from `LD_LIBRARY_PATH`.
pub fn remove_from_ld_library_path(path: &str) {
    let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let entries: Vec<&str> = current.split(':').filter(|e| *e != path).collect();
    let new_path = entries.join(":");
    unsafe {
        env::set_var("LD_LIBRARY_PATH", &new_path);
    }
}

/// Get all entries in `LD_LIBRARY_PATH`.
pub fn ld_library_path_entries() -> Vec<String> {
    let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if current.is_empty() {
        return Vec::new();
    }
    current.split(':').map(String::from).collect()
}

/// Check if a shared library exists in `LD_LIBRARY_PATH`.
///
/// Returns the full path if found, `None` otherwise.
pub fn find_library(name: &str) -> Option<String> {
    for dir in ld_library_path_entries() {
        let candidate = format!("{}/{}", dir, name);
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    None
}

/// Ensure common UCX/PMIx library directories are in `LD_LIBRARY_PATH`.
///
/// Checks typical install locations: `/usr/local/lib`, `/usr/lib`,
/// and any paths specified in `UCX_HOME` or `PMIX_HOME`.
pub fn ensure_hpc_lib_paths() {
    // Standard system paths
    for path in &[
        "/usr/local/lib",
        "/usr/lib",
        "/usr/local/lib64",
        "/usr/lib64",
    ] {
        if std::path::Path::new(path).exists() {
            add_to_ld_library_path(path);
        }
    }

    // UCX_HOME
    if let Ok(ucx_home) = env::var("UCX_HOME") {
        add_to_ld_library_path(&format!("{}/lib", ucx_home));
        add_to_ld_library_path(&format!("{}/lib64", ucx_home));
    }

    // PMIX_HOME
    if let Ok(pmix_home) = env::var("PMIX_HOME") {
        add_to_ld_library_path(&format!("{}/lib", pmix_home));
        add_to_ld_library_path(&format!("{}/lib64", pmix_home));
    }

    // PREFIX (common CMake variable)
    if let Ok(prefix) = env::var("PREFIX") {
        add_to_ld_library_path(&format!("{}/lib", prefix));
        add_to_ld_library_path(&format!("{}/lib64", prefix));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_to_ld_library_path() {
        unsafe {
            env::remove_var("LD_LIBRARY_PATH");
        }
        let added = add_to_ld_library_path("/test/path");
        assert!(added);
        let entries = ld_library_path_entries();
        assert!(entries.contains(&"/test/path".to_string()));
    }

    #[test]
    fn test_add_duplicate() {
        unsafe {
            env::set_var("LD_LIBRARY_PATH", "/test/path");
        }
        let added = add_to_ld_library_path("/test/path");
        assert!(!added);
    }

    #[test]
    fn test_ld_library_path_entries() {
        unsafe {
            env::set_var("LD_LIBRARY_PATH", "/a:/b:/c");
        }
        let entries = ld_library_path_entries();
        assert_eq!(entries, vec!["/a", "/b", "/c"]);
    }
}
