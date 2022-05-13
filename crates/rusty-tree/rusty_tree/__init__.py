"""
RustyTree: Large scale distributed octrees build with Rust, exposed via a
Python interface.
"""
from .rusty_tree import lib, ffi
from rusty_tree.distributed import DistributedTree
