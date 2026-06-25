//! High-resolution timing utilities.
//!
//! Provides wall-clock timing and microsecond-precision measurements
//! matching the C reference `MPI_Wtime()` behavior.

#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

/// Get current time in microseconds since an arbitrary epoch.
///
/// Uses `Instant::now()` which is monotonic and high-resolution.
/// The returned value is relative to process start.
pub fn now_us() -> f64 {
    lazy_static::init();
    INSTANT_NOW
        .get()
        .map(|i| i.elapsed().as_secs_f64() * 1_000_000.0)
        .unwrap_or(0.0)
}

/// Wall-clock timer wrapper matching C `MPI_Wtime()`.
///
/// Returns seconds (f64) since process start, suitable for
/// computing elapsed time as `t_end - t_start`.
#[derive(Debug, Clone, Copy)]
pub struct Wtime {
    instant: Instant,
}

impl Wtime {
    /// Create a new timer anchored to `Instant::now()`.
    pub fn new() -> Self {
        Self {
            instant: Instant::now(),
        }
    }

    /// Get elapsed seconds since construction.
    pub fn elapsed(&self) -> f64 {
        self.instant.elapsed().as_secs_f64()
    }

    /// Get elapsed microseconds since construction.
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed() * 1_000_000.0
    }

    /// Get elapsed milliseconds since construction.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed() * 1_000.0
    }

    /// Measure a closure and return elapsed time in microseconds.
    pub fn measure_us<F, R>(f: F) -> (R, f64)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        (result, elapsed)
    }

    /// Measure a closure and return elapsed time in seconds.
    pub fn measure<F, R>(f: F) -> (R, f64)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_secs_f64();
        (result, elapsed)
    }
}

impl Default for Wtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Global instant for `now_us()` baseline.
mod lazy_static {
    use std::sync::OnceLock;
    use std::time::Instant;

    pub static INSTANT_NOW: OnceLock<Instant> = OnceLock::new();

    pub fn init() {
        INSTANT_NOW.get_or_init(Instant::now);
    }
}

/// Global instant for `now_us()` baseline (accessible from parent module).
use lazy_static::INSTANT_NOW;

/// Timing result for a benchmark iteration.
#[derive(Debug, Clone, Copy)]
pub struct TimingResult {
    /// Minimum time in microseconds.
    pub min_us: f64,
    /// Maximum time in microseconds.
    pub max_us: f64,
    /// Average time in microseconds.
    pub avg_us: f64,
    /// Total accumulated time in microseconds.
    pub total_us: f64,
    /// Number of timed iterations.
    pub count: usize,
}

impl TimingResult {
    /// Create a new empty timing result.
    pub fn new() -> Self {
        Self {
            min_us: f64::MAX,
            max_us: f64::MIN,
            avg_us: 0.0,
            total_us: 0.0,
            count: 0,
        }
    }

    /// Add a new timing sample (in microseconds).
    pub fn add(&mut self, us: f64) {
        self.min_us = self.min_us.min(us);
        self.max_us = self.max_us.max(us);
        self.total_us += us;
        self.count += 1;
        self.avg_us = self.total_us / self.count as f64;
    }

    /// Get the average time, or 0.0 if no samples.
    pub fn average(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.avg_us }
    }
}

impl Default for TimingResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wtime_elapsed() {
        let timer = Wtime::new();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 9.0);
    }

    #[test]
    fn test_timing_result() {
        let mut result = TimingResult::new();
        result.add(100.0);
        result.add(200.0);
        result.add(300.0);
        assert_eq!(result.min_us, 100.0);
        assert_eq!(result.max_us, 300.0);
        assert_eq!(result.avg_us, 200.0);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_now_us() {
        let t1 = now_us();
        std::thread::sleep(Duration::from_millis(1));
        let t2 = now_us();
        assert!(t2 > t1);
    }
}
