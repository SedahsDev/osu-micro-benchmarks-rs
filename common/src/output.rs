//! Output formatting matching the C reference implementation.
//!
//! The C reference uses `FIELD_WIDTH = 20` and `FLOAT_PRECISION = 2`.
//! Column headers are printed with `#` prefix. Data rows use fixed-width
//! formatting: `{:<10}` for size, `{:>20.2}` for values.

use std::io::Write;

/// Field width for output columns (matches C FIELD_WIDTH).
pub const FIELD_WIDTH: usize = 20;
/// Float precision for output values (matches C FLOAT_PRECISION).
pub const FLOAT_PRECISION: usize = 2;
/// Size column width.
pub const SIZE_WIDTH: usize = 10;
/// OMB version string.
pub const VERSION: &str = "7.5.2";

/// Benchmark output type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchmarkType {
    /// Point-to-point latency test.
    Latency,
    /// Point-to-point bandwidth test.
    Bandwidth,
    /// Bidirectional bandwidth test.
    BiBandwidth,
    /// Multi-buffer multi-region bandwidth.
    MbwMr,
    /// Collective latency test.
    CollectiveLatency,
    /// Collective bandwidth test.
    CollectiveBandwidth,
    /// Non-blocking collective test.
    NonBlockingCollective,
    /// Startup test.
    Startup,
    /// Congestion bandwidth test.
    CongestionBw,
}

/// Print the benchmark header line.
///
/// Format: `# OSU MPI <Name> Test (v7.5.2)`
pub fn print_header<W: Write>(writer: &mut W, name: &str, bench_type: BenchmarkType) {
    let title = match bench_type {
        BenchmarkType::Latency => format!("OSU MPI Latency Test (v{})", VERSION),
        BenchmarkType::Bandwidth => format!("OSU MPI Bandwidth Test (v{})", VERSION),
        BenchmarkType::BiBandwidth => format!("OSU MPI Bidirectional Bandwidth Test (v{})", VERSION),
        BenchmarkType::MbwMr => format!("OSU MPI Multi-Buffer and Multi-Region Bandwidth Test (v{})", VERSION),
        BenchmarkType::CollectiveLatency => format!("OSU MPI Collective Latency Test (v{})", VERSION),
        BenchmarkType::CollectiveBandwidth => format!("OSU MPI Collective Bandwidth Test (v{})", VERSION),
        BenchmarkType::NonBlockingCollective => format!("OSU MPI Non-Blocking Collective Latency Test (v{})", VERSION),
        BenchmarkType::Startup => format!("OSU MPI Startup Test (v{})", VERSION),
        BenchmarkType::CongestionBw => format!("OSU MPI Congestion Bandwidth Test (v{})", VERSION),
    };
    let _ = writeln!(writer, "# {}", title);
    let _ = writer.flush();
    let _ = name; // used for benchmark_name tracking
}

/// Print the column header for latency benchmarks.
///
/// Format: `# Size               Latency (usec)`
pub fn print_latency_header<W: Write>(writer: &mut W) {
    let _ = writeln!(writer, "# {:<10} {:>20}", "Size", "Latency (usec)");
    let _ = writer.flush();
}

/// Print the column header for bandwidth benchmarks.
///
/// Format: `# Size              Bandwidth (MB/s)`
pub fn print_bandwidth_header<W: Write>(writer: &mut W) {
    let _ = writeln!(writer, "# {:<10} {:>20}", "Size", "Bandwidth (MB/s)");
    let _ = writer.flush();
}

/// Print the column header for bidirectional bandwidth benchmarks.
///
/// Format: `# Size              BW (MB/s)      Message Rate (Mmsg/s)`
pub fn print_bibw_header<W: Write>(writer: &mut W) {
    let _ = writeln!(
        writer,
        "# {:<10} {:>20} {:>23}",
        "Size", "BW (MB/s)", "Message Rate (Mmsg/s)"
    );
    let _ = writer.flush();
}

/// Print a latency data row.
///
/// Format: `<size>    <avg>    <min>    <max>`
/// Uses C-matching format: `{:<10}` for size, `{:>20.2}` for latency values.
pub fn print_latency_row<W: Write>(writer: &mut W, size: usize, avg: f64, min: f64, max: f64) {
    let _ = write!(writer, "{:<10} {:>20.2} {:>20.2} {:>20.2}", size, avg, min, max);
    let _ = writer.flush();
}

/// Print a latency data row (average only).
///
/// Format: `<size>    <avg>`
pub fn print_latency_avg<W: Write>(writer: &mut W, size: usize, avg: f64) {
    let _ = write!(writer, "{:<10} {:>20.2}", size, avg);
    let _ = writer.flush();
}

/// Print a bandwidth data row.
///
/// Format: `<size>    <avg>    <min>    <max>`
pub fn print_bandwidth_row<W: Write>(writer: &mut W, size: usize, avg: f64, min: f64, max: f64) {
    let _ = write!(writer, "{:<10} {:>20.2} {:>20.2} {:>20.2}", size, avg, min, max);
    let _ = writer.flush();
}

/// Print a bandwidth data row (average only).
pub fn print_bandwidth_avg<W: Write>(writer: &mut W, size: usize, avg: f64) {
    let _ = write!(writer, "{:<10} {:>20.2}", size, avg);
    let _ = writer.flush();
}

/// Print a newline after a data row.
pub fn print_newline<W: Write>(writer: &mut W) {
    let _ = writeln!(writer);
    let _ = writer.flush();
}

/// Print a latency data row with validation status.
pub fn print_latency_row_validate<W: Write>(
    writer: &mut W,
    size: usize,
    avg: f64,
    errors: usize,
) {
    let status = if errors > 0 { "Fail" } else { "Pass" };
    let _ = write!(
        writer,
        "{:<10} {:>20.2} {:>20}",
        size, avg, status
    );
    let _ = writer.flush();
}

/// Print the non-blocking collective header.
///
/// Format: `# Size  Overlap  CPU  Communication  Wait  Init`
pub fn print_nbc_header<W: Write>(writer: &mut W) {
    let _ = writeln!(
        writer,
        "# {:<10} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "Size", "Overlap", "CPU", "Communication", "Wait", "Init"
    );
    let _ = writer.flush();
}

/// Print a non-blocking collective data row.
///
/// Format: `<size>    <overlap>    <cpu>    <communication>    <wait>    <init>`
pub fn print_nbc_row<W: Write>(
    writer: &mut W,
    size: usize,
    overlap: f64,
    cpu: f64,
    communication: f64,
    wait: f64,
    init: f64,
) {
    let _ = write!(
        writer,
        "{:<10} {:>20.2} {:>20.2} {:>20.2} {:>20.2} {:>20.2}",
        size, overlap, cpu, communication, wait, init
    );
    let _ = writer.flush();
}

/// Print startup test output.
pub fn print_startup<W: Write>(writer: &mut W, label: &str, time_ms: f64) {
    let _ = writeln!(writer, "{:<30} {:>12.4} ms", label, time_ms);
    let _ = writer.flush();
}

/// Print a newline-terminated data row for simple output.
pub fn print_data_line<W: Write>(writer: &mut W, size: usize, value: f64) {
    let _ = writeln!(writer, "{:<10} {:>20.2}", size, value);
    let _ = writer.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_header() {
        let mut buf = Vec::new();
        print_latency_header(&mut buf);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Size"));
        assert!(output.contains("Latency (usec)"));
    }

    #[test]
    fn test_bandwidth_header() {
        let mut buf = Vec::new();
        print_bandwidth_header(&mut buf);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Size"));
        assert!(output.contains("Bandwidth (MB/s)"));
    }

    #[test]
    fn test_latency_row_format() {
        let mut buf = Vec::new();
        print_latency_avg(&mut buf, 65536, 1.234);
        print_newline(&mut buf);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("65536"));
        assert!(output.contains("1.23"));
    }
}
