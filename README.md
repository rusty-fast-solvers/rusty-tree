# Rusty Tree

[![Anaconda-Server Badge](https://anaconda.org/skailasa/rusty_tree/badges/platforms.svg)](https://anaconda.org/skailasa/rusty_tree) [![Anaconda-Server Badge](https://anaconda.org/skailasa/rusty_tree/badges/latest_release_date.svg)](https://anaconda.org/skailasa/rusty_tree) [![Anaconda-Server Badge](https://anaconda.org/skailasa/rusty_tree/badges/version.svg)](https://anaconda.org/skailasa/rusty_tree)

Implementation of distributed Octrees in Rust with Python interfaces for
Scientific Computing.

Usage examples and build instructions can be found in the
[project wiki](https://github.com/rusty-fast-solvers/rusty-tree/wiki).

## Install

The Conda package installs all required dependencies including MPI. Installing
only the Rust package relies on a correctly configured MPI installation on your
system. Specifically, it should support the [`mpicc -show`](https://github.com/rsmpi/rsmpi)
command

### Python

Install Python package from Anaconda Cloud into a Conda environment.

```bash
conda install -c skailasa rusty_tree
```

Install mpi4py dependency separately, to ensure that correct pointers are
created to shared libraries when installing into a virtual environment

```
env MPICC=/path/to/mpicc python -m pip install mpi4py
```

**Installation requires that your channel list contains `conda-forge`.**=

### Rust

The Rust library can be added to your project's `Cargo.toml` from source.

```toml
rusty-tree = { git = "https://github.com/rusty-fast-solvers/rusty-tree", branch = "main"}
```
