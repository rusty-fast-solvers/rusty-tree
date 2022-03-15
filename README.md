# Rusty Tree

Implementation of Octrees [1] in Rust with Python interfaces.

## Build

```bash
# Build crates
cargo build
```

## Usage

...


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

