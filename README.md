# Rusty Tree

Implementation of Octrees [1] in Rust with Python interfaces.

# Python Library

Install and use from Anaconda, relies on a working MPI implementation on your system.

```bash
conda install -c skailasa rusty_tree
```

## Build

```bash
cd crates/rusty-tree/ && maturin develop --release
```

## Usage

Write a script:

```python
from mpi4py import MPI
import numpy as np

from rusty_tree.distributed import DistributedTree


# Setup communicator
comm = MPI.COMM_WORLD

# Cartesian points at this process
points = np.random.rand(100000, 3)

# Generate a balanced tree
balanced = DistributedTree.from_global_points(points, True, comm)

# Generate an unbalanced tree
unbalanced = DistributedTree.from_global_points(points, False, comm)
```

Run a script using mpi4py (specified in requirements)

```bash
mpiexec -n <nprocs> python -m mpi4py /path/to/script/
```

# Rust Library

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


fn main () {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points: Vec<[PointType; 3]> = Vec::new();
    let npoints = 1000000;

    for _ in 0..npoints {
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

    // Unbalanced
    let unbalanced: DistributedTree = DistributedTree::new(&points, false, &comm)

    // Balanced
    let balanced_tree: DistributedTree = DistributedTree::new(&points, true, &comm)
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
