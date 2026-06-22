//! Helper functions for the runtime module.

use ucx_sys::ep;
use ucx_sys::worker;

/// Flush an endpoint by flushing the worker.
pub fn flush_ep_blocking(worker: &worker::Worker, _ep: &ep::Ep, param: &ucx_sys::RequestParam) {
    // Worker.flush() flushes all AM/RMA on this worker.
    let req = worker.flush(param);
    if let Ok(Some(r)) = req {
        while !r.check_finished().unwrap_or(false) {
            loop {
                if !worker.progress() {
                    break;
                }
            }
        }
    }
}
