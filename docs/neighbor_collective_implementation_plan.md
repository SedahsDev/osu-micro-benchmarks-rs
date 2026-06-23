# Implementation Plan: MPI Neighbor Collectives (Rust)

## 1. Background & Context

### What are neighbor collectives?
Neighbor collectives operate on a **subset of processes** defined by a graph topology (`MPI_Dist_graph`), rather than all processes in the communicator. Each process has an *indegree* (incoming neighbors) and *outdegree* (outgoing neighbors) with explicit neighbor lists.

### C reference pattern (all 5 files share this structure):
```
1. Parse CLI args (including `-N` neighborhood config)
2. Call omb_neighborhood_create() → builds indegree/sources/outdegree/destinations
3. MPI_Dist_graph_create_adjacent() → creates dist_graph communicator
4. Allocate send/recv buffers sized for neighbor count
5. Run benchmark loop using the dist_graph communicator
6. MPI_Comm_free(dist_graph_comm)
```

### Neighborhood configuration (`-N` flag):
The C reference supports three topology types:
- **`cart`** (default): n-dimensional cartesian grid (e.g., `2x4x2`) — each rank connected to its grid neighbors
- **`star`**: star topology — rank 0 connected to all others
- **`graph`**: user-provided adjacency list file (CSV: `src,dst` per line)

The `omb_neighborhood_create()` function in `osu_util_mpi.c` (line 1507+) computes the indegree, sources, sourceweights, outdegree, destinations, destweights arrays that feed into `MPI_Dist_graph_create_adjacent`.

---

## 2. Current Rust Codebase Status

### What exists:
- **`common/src/runtime/context.rs`**: `OsUContext` with PMIx + UCX + UCC, endpoints per rank
- **`common/src/runtime/non_blocking.rs`**: Non-blocking collective implementations using UCX tag-matching (iallgather, iallgatherv, ialltoall, ialltoallv, ialltoallw, ibcast, igather, igatherv, ireduce, ireduce_scatter, ireduce_scatter_block, iscatter, iscatterv) — all operate on **full communicator**
- **`common/src/cli.rs`**: CLI args including `neighborhood: Option<String>` (line 202-203)
- **`collective/src/bin/osu_ineighbor_*.rs`**: 5 stub files that just print "TODO: Implement" and exit

### What's missing:
- Neighborhood topology builder (equivalent to `omb_neighborhood_create`)
- Dist-graph communicator abstraction
- Neighbor-scoped non-blocking collective operations in `non_blocking.rs`
- Actual benchmark implementations in the 5 stub binaries

---

## 3. Detailed Implementation Plan

### Phase 1: Neighborhood Topology Builder

**New file:** `common/src/runtime/neighborhood.rs`

This module provides the Rust equivalent of `omb_neighborhood_create()`.

#### 3.1 Data Structures

```rust
/// Neighborhood topology type
#[derive(Debug, Clone)]
pub enum NeighborhoodType {
    /// N-dimensional cartesian grid (e.g., "2x4x2")
    Cart(Vec<usize>),
    /// Star topology (rank 0 connected to all others)
    Star,
    /// User-provided adjacency list from file
    Graph { filepath: String },
}

/// Computed neighborhood topology for a single rank
#[derive(Debug, Clone)]
pub struct NeighborhoodTopology {
    /// Number of incoming neighbors
    pub indegree: usize,
    /// Ranks that send TO this rank
    pub sources: Vec<usize>,
    /// Weights for incoming edges (always 1 in OSU benchmarks)
    pub sourceweights: Vec<usize>,
    /// Number of outgoing neighbors
    pub outdegree: usize,
    /// Ranks this rank sends TO
    pub destinations: Vec<usize>,
    /// Weights for outgoing edges (always 1 in OSU benchmarks)
    pub destweights: Vec<usize>,
}
```

#### 3.2 Parsing the `-N` flag

```rust
impl NeighborhoodType {
    /// Parse neighborhood string from CLI: "cart:2x4", "star", "graph:/path/to/file"
    pub fn from_cli(s: &str) -> Result<Self, String> {
        // "cart:dims" → Cart
        // "star" → Star  
        // "graph:filepath" → Graph
    }
}
```

#### 3.3 Topology computation functions

```rust
/// Build neighborhood topology for cartesian grid
pub fn build_cart_neighborhood(
    dims: &[usize],
    rank: usize,
    size: usize,
) -> NeighborhoodTopology { ... }

/// Build neighborhood topology for star
pub fn build_star_neighborhood(
    rank: usize,
    size: usize,
) -> NeighborhoodTopology { ... }

/// Build neighborhood topology from adjacency file
pub fn build_graph_neighborhood(
    filepath: &str,
    rank: usize,
    size: usize,
) -> NeighborhoodTopology { ... }

/// Unified entry point (equivalent to omb_neighborhood_create)
pub fn create_neighborhood(
    topo_type: &NeighborhoodType,
    rank: usize,
    size: usize,
) -> NeighborhoodTopology { ... }
```

**Cartesian grid logic** (from C reference lines 1600+):
- Compute coordinates for each rank in the N-dimensional grid
- For each dimension, check if there's a neighbor in +1 and -1 direction
- Non-periodic (periods = 0 for all dims)
- Collect neighbor ranks as sources and destinations
- indegree == outdegree for symmetric cartesian grids

**Star topology** (from C reference):
- Rank 0: outdegree = size-1 (connected to all), indegree = size-1
- Rank N (N>0): outdegree = 1 (connected to rank 0), indegree = 1

**Graph from file** (from C reference lines 1528+):
- Read CSV file with `src,dst` pairs
- Filter pairs where src == my_rank → destinations
- Filter pairs where dst == my_rank → sources
- Need PMIx barrier/fence to ensure all ranks read before proceeding

#### 3.4 Broadcasting neighborhood info

Since each rank computes its own indegree/sources/outdegree/destinations, we need to ensure consistency. For the `graph` type, we need to exchange neighbor lists via PMIx or UCX. The C reference uses `MPI_Allgather` within the original communicator before creating the dist_graph.

In our Rust implementation, we can use the existing `ctx.barrier()` and exchange neighbor lists via the existing UCX endpoints or PMIx.

---

### Phase 2: Dist-Graph Communicator Abstraction

**New file:** `common/src/runtime/dist_graph.rs`

This module wraps the concept of a dist-graph communicator. Since we don't have MPI, we simulate it by maintaining the neighborhood topology and routing messages only to/from neighbors.

```rust
pub struct DistGraphContext {
    /// Underlying full-communicator context
    pub base_ctx: OsUContext,
    /// Neighborhood topology for this rank
    pub topology: NeighborhoodTopology,
}

impl DistGraphContext {
    /// Create dist-graph context from neighborhood config
    pub fn create(
        base_ctx: &OsUContext,
        neighborhood: &NeighborhoodType,
    ) -> Self { ... }
    
    /// Get the number of neighbors (indegree)
    pub fn num_neighbors(&self) -> usize { self.topology.indegree }
    
    /// Get neighbor ranks (sources)
    pub fn neighbors(&self) -> &[usize] { &self.topology.sources }
    
    /// Get destination ranks
    pub fn destinations(&self) -> &[usize] { &self.topology.destinations }
}
```

---

### Phase 3: Neighbor Non-Blocking Collective Operations

**Extend:** `common/src/runtime/non_blocking.rs`

Add neighbor-scoped versions of the 5 collective operations.

#### 3.5 Neighbor Allgather (`ineighbor_allgather`)

**C pattern:** Each rank sends `msg_size` bytes to each neighbor. Receives `msg_size` bytes from each neighbor. Total recv buffer = `indegree * msg_size`.

**Rust implementation:**
```rust
impl OsUContext {
    /// Non-blocking neighbor allgather
    /// Each neighbor sends `msg_size` bytes, recv buffer holds `indegree * msg_size` bytes
    pub fn ineighbor_allgather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
        neighbors: &[usize],
    ) -> OsURequest { ... }
}
```

**Algorithm:**
1. Post `indegree` non-blocking receives (one per neighbor), each receiving `msg_size` bytes
   - Tag = BASE_TAG + rank (unique per sender)
   - Place received data at offset `i * msg_size` in recvbuf
2. Post `outdegree` non-blocking sends (one per destination), each sending `msg_size` bytes
   - Tag = BASE_TAG + my_rank
3. Return combined request wrapping all send+recv UCX requests

**Key difference from regular iallgather:**
- Regular: communicate with ALL ranks (size sends + size receives)
- Neighbor: communicate only with neighbors (indegree receives + outdegree sends)

#### 3.6 Neighbor Allgatherv (`ineighbor_allgatherv`)

**C pattern:** Variable-size contributions. Each neighbor sends a different amount. Uses `recvcounts` array specifying how many bytes to expect from each neighbor, and `displs` for placement offsets.

**Rust implementation:**
```rust
impl OsUContext {
    pub fn ineighbor_allgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        sendcounts: &[usize],  // bytes to send to each destination
        recvcounts: &[usize],  // bytes to receive from each source
        displs: &[usize],      // displacement in recvbuf for each source's data
        neighbors: &[usize],
        destinations: &[usize],
    ) -> OsURequest { ... }
}
```

**Algorithm:**
1. Post `indegree` non-blocking receives with variable counts and displacements
2. Post `outdegree` non-blocking sends with variable counts
3. Return combined request

**Buffer layout from C reference:**
- `recvcounts[i]` = msg_size (same for all in the benchmark, but API supports variable)
- `displs[i]` = i * msg_size

#### 3.7 Neighbor Alltoall (`ineighbor_alltoall`)

**C pattern:** Each rank sends `msg_size` bytes to each neighbor and receives `msg_size` bytes from each neighbor. Symmetric exchange.

**Rust implementation:**
```rust
impl OsUContext {
    pub fn ineighbor_alltoall(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        msg_size: usize,
        neighbors: &[usize],
    ) -> OsURequest { ... }
}
```

**Algorithm:**
1. Send `msg_size` bytes to each neighbor (data from `sendbuf[i * msg_size]`)
2. Receive `msg_size` bytes from each neighbor (data to `recvbuf[i * msg_size]`)
3. Return combined request

**Note:** This is functionally identical to neighbor_allgather for symmetric topologies, but the semantic meaning differs. The C reference treats them the same way.

#### 3.8 Neighbor Alltoallv (`ineighbor_alltoallv`)

**C pattern:** Variable-size all-to-all with neighbors. Uses `sdispls`, `sendcounts`, `rdispls`, `recvcounts`.

**Rust implementation:**
```rust
impl OsUContext {
    pub fn ineighbor_alltoallv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        sendcounts: &[usize],
        sdispls: &[usize],
        recvcounts: &[usize],
        rdispls: &[usize],
        neighbors: &[usize],
    ) -> OsURequest { ... }
}
```

**Algorithm:**
1. For each destination `i`, send `sendcounts[i]` bytes from `sendbuf[sdispls[i]..]`
2. For each source `i`, receive `recvcounts[i]` bytes into `recvbuf[rdispls[i]..]`
3. Return combined request

**Buffer setup from C reference:**
- `sendcounts[i] = msg_size`, `sdispls[i] = i * msg_size`
- `recvcounts[i] = msg_size`, `rdispls[i] = i * msg_size`

#### 3.9 Neighbor Alltoallw (`ineighbor_alltoallw`)

**C pattern:** Weighed all-to-all — the most general variant. Each send/recv can have different counts, displacements, AND data types. Uses `sendtypes` and `recvtypes` arrays.

**Rust implementation:**
```rust
impl OsUContext {
    pub fn ineighbor_alltoallw(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        senddescriptions: &[(usize, usize, usize)],  // (count, disp, type_size) per destination
        recvdescriptions: &[(usize, usize, usize)],  // (count, disp, type_size) per source
        neighbors: &[usize],
    ) -> OsURequest { ... }
}
```

**Algorithm:**
1. For each destination `i`, send based on `senddescriptions[i]`
2. For each source `i`, receive based on `recvdescriptions[i]`
3. Return combined request

**Data type handling:** The C reference uses `MPI_Type_create_struct` for the send/recv types. In our UCX-based implementation, we handle this by computing byte offsets and sizes directly (since UCX tag matching works at the byte level).

---

### Phase 4: Benchmark Binaries

**Files to implement:**
- `collective/src/bin/osu_ineighbor_allgather.rs`
- `collective/src/bin/osu_ineighbor_allgatherv.rs`
- `collective/src/bin/osu_ineighbor_alltoall.rs`
- `collective/src/bin/osu_ineighbor_alltoallv.rs`
- `collective/src/bin/osu_ineighbor_alltoallw.rs`

#### Common benchmark structure (pattern from C reference):

```rust
fn run_benchmark(ctx: &OsUContext, args: &CliArgs) {
    let rank = ctx.rank();
    let size = ctx.size();
    
    // 1. Parse neighborhood config
    let neighborhood_type = NeighborhoodType::from_cli(
        args.neighborhood.as_deref().unwrap_or("cart:2")
    ).expect("invalid neighborhood");
    
    // 2. Create dist-graph context
    let dg_ctx = DistGraphContext::create(ctx, &neighborhood_type);
    let neighbors = dg_ctx.neighbors();
    let num_neighbors = dg_ctx.num_neighbors();
    
    // 3. Barrier
    ctx.barrier();
    
    // 4. Setup message sizes (0 for latency, or iterate for bandwidth)
    //    The C reference uses fixed msg_size=0 for latency tests
    
    // 5. Allocate buffers
    let msg_size = 0;  // latency test
    let sendbuf = vec![0u8; msg_size * num_neighbors];
    let recvbuf = vec![0u8; msg_size * num_neighbors];
    
    // 6. Run benchmark loop
    for i in 0..(iterations + skip) {
        let t_start = Wtime::new();
        
        let init_start = Wtime::new();
        let mut request = ctx.ineighbor_allgather(&sendbuf, &mut recvbuf, msg_size, neighbors);
        let init_time = init_start.elapsed_us();
        
        let comp_time = 0.0;  // no dummy compute for neighbor collectives
        
        let wait_start = Wtime::new();
        request.wait();
        let wait_time = wait_start.elapsed_us();
        
        let elapsed_us = t_start.elapsed_us();
        
        if i >= skip {
            // Accumulate timing
            // Barrier between iterations
            ctx.barrier();
        }
    }
    
    // 7. Print results (NBC format)
    if rank == 0 {
        output::print_nbc_header(&mut out);
        output::print_nbc_row(&mut out, msg_size, overlap, cpu, comm, wait, init);
    }
}
```

#### Per-benchmark differences:

| Benchmark | Send pattern | Recv pattern | Extra arrays |
|-----------|-------------|-------------|-------------|
| `ineighbor_allgather` | `msg_size` to each dest | `msg_size` from each source | None |
| `ineighbor_allgatherv` | `sendcounts[i]` to each dest | `recvcounts[i]` from each source | `recvcounts`, `displs` |
| `ineighbor_alltoall` | `msg_size` to each neighbor | `msg_size` from each neighbor | None |
| `ineighbor_alltoallv` | `sendcounts[i]` from `sdispls[i]` | `recvcounts[i]` to `rdispls[i]` | `sendcounts`, `sdispls`, `recvcounts`, `rdispls` |
| `ineighbor_alltoallw` | Per-dest (count, disp, type) | Per-source (count, disp, type) | Full descriptions |

---

## 4. Module Structure Changes

### New files to create:
```
common/src/runtime/
├── mod.rs              # Add: mod neighborhood; mod dist_graph;
├── neighborhood.rs     # NEW: NeighborhoodType, topology builders
├── dist_graph.rs       # NEW: DistGraphContext wrapper
└── non_blocking.rs     # EXTEND: Add 5 neighbor collective methods
```

### Updated exports in `mod.rs`:
```rust
mod collective_blocking;
mod constants;
mod context;
mod dist_graph;         // NEW
mod helpers;
mod neighborhood;       // NEW
mod non_blocking;
mod ucc_oob;

pub use context::OsUContext;
pub use dist_graph::DistGraphContext;  // NEW
pub use neighborhood::NeighborhoodType; // NEW
pub use non_blocking::OsURequest;
```

### Binary files to implement:
```
collective/src/bin/
├── osu_ineighbor_allgather.rs    # Implement (currently stub)
├── osu_ineighbor_allgatherv.rs   # Implement (currently stub)
├── osu_ineighbor_alltoall.rs     # Implement (currently stub)
├── osu_ineighbor_alltoallv.rs    # Implement (currently stub)
└── osu_ineighbor_alltoallw.rs    # Implement (currently stub)
```

---

## 5. Implementation Order (Recommended)

1. **`neighborhood.rs`** — Topology builder (no runtime dependency, pure computation)
   - Start with `cart` topology (most commonly used)
   - Add `star` topology
   - Add `graph` file-based topology

2. **`dist_graph.rs`** — Thin wrapper combining OsUContext + NeighborhoodTopology

3. **`non_blocking.rs` extension** — Add `ineighbor_allgather` first (simplest, symmetric)
   - Then `ineighbor_alltoall` (nearly identical)
   - Then `ineighbor_allgatherv` (adds recvcounts/displs)
   - Then `ineighbor_alltoallv` (adds sendcounts/sdispls)
   - Finally `ineighbor_alltoallw` (most complex, per-operation type info)

4. **Benchmark binaries** — Implement one at a time, testing each before moving on
   - Start with `osu_ineighbor_allgather` (simplest)
   - Each follows the same pattern, just calling different runtime methods

---

## 6. Key Technical Details

### Tag management for neighbor collectives
The existing `non_blocking.rs` uses `BASE_TAG` constants. Neighbor collectives need unique tags per sender to distinguish incoming messages from different neighbors. Strategy:
- Tag = `BASE_TAG_NBR + sender_rank` (unique per sender)
- Receiver probes on specific tags from known neighbor ranks

### Buffer management
- Send buffers: `outdegree × msg_size` bytes (one chunk per destination)
- Recv buffers: `indegree × msg_size` bytes (one chunk per source)
- For variable-size variants, use displacements to index into buffers

### Request combining
The existing `OsURequest` wraps a single UCX request. For neighbor collectives that involve multiple sends+receives, we need to either:
- **Option A:** Create a `Vec<OsURequest>` and wait on all
- **Option B:** Create a new `OsURequest::Group(Vec<OsURequest>)` variant
- **Recommendation:** Option B — extends the existing enum cleanly and maintains the `wait()` API

### PMIx synchronization for graph topology
When using file-based graph topology, all ranks need to agree on the topology before proceeding. Use the existing PMIx fence/barrier mechanism.

### Handling asymmetric topologies
Some topologies have different indegree and outdegree. The implementation must handle:
- `indegree != outdegree` (star topology: center has high degree, leaves have degree 1)
- `sources != destinations` (directed graphs)

### Data validation
The C reference validates data after the collective completes. For neighbor collectives, validation checks that:
- Each neighbor's contribution is correctly placed in the recv buffer
- Data sent by rank X is received by all of X's destinations

---

## 7. Testing Strategy

1. **Unit tests** for `neighborhood.rs`:
   - Test cartesian grid with various dimensions (2, 2x2, 2x3x4)
   - Test star topology with various sizes
   - Verify indegree/sources/outdegree/destinations are correct

2. **Integration tests** for each benchmark:
   - Run with 2, 4, 8 processes
   - Test default cartesian topology
   - Test star topology
   - Verify output format matches NBC pattern

3. **Correctness validation**:
   - Enable `--validate` flag
   - Verify recv buffer contents match expected pattern

---

## 8. Potential Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| Tag collisions between neighbor and non-neighbor collectives | Use separate BASE_TAG ranges (e.g., `BASE_TAG_NBR = 0x1000`) |
| Large neighbor counts causing too many UCX requests | Batch sends/receives or use request pooling |
| Graph file parsing edge cases | Validate file format, check rank bounds, handle MPI_UNDEFINED |
| Asymmetric topologies with mismatched indegree/outdegree | Ensure send side uses destinations, recv side uses sources |
| PMIx fence ordering for graph topology | Use existing `ctx.barrier()` before dist-graph creation |
