# Rusty Tree

Implementation of Octrees [1] in Rust with Python interfaces.

## Build

```bash
# Build crates
cargo build
```

## Usage

```rust
use rand::prelude::*;
use rand::SeedableRng;

use mpi::{
    environment::Universe, 
    topology::{Color, UserCommunicator},
    traits::*
};

use rusty_tree::{
    constants::{NCRIT, ROOT},
    distributed::DistributedTree,
    types::{
        domain::Domain,
        morton::MortonKey
        point::PointType, 
    },
};

const NPOINTS: u64 = 100000;

fn main () {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points: Vec<[PointType; 3]> = Vec::new();

    for _ in 0..NPOINTS {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    }

    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let comm: UserCommunicator = universe.world();
    let comm = comm.split_by_color(Color::with_value(0)).unwrap();
    
    // Calculate the global domain defined by the distributed points
    let domain: Domain = compute_global_domain(&points, &comm);

    // Generate a distributed tree

    // Unbalanced
    let unbalanced: DistributedTree = DistributedTree::new(&points, &domain, false, &universe)

    // Balanced
    let balanced_tree: DistributedTree = DistributedTree::new(&points, &domain, true, &universe)
}
```

## Test

Tests for serial code managed by Cargo

```bash
cargo test
```

Parallel functionality is tested via the binary `parallel-tests` crate.

```bash
mpirun -n <nprocs> ./target/<release/debug>/parallel-tests
```

## Citation

```bash
@software{rusty-tree,
  author = {{Timo Betcke, Srinath Kailasa}},
  title = {Rusty Tree,
  url = {https://github.com/rusty-fast-solveres/rusty-tree},
  version = {0.1.0},
  date = {2022-03-20},
}
```

### References

[1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5 (2008): 2675-2708.

[2] Sundar, Hari, Dhairya Malhotra, and George Biros. "Hyksort: a new variant of hypercube quicksort on distributed memory architectures." Proceedings of the 27th international ACM conference on international conference on supercomputing. 2013.
