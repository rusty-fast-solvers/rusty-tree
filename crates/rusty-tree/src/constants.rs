//! Crate wide constants
use mpi::topology::Rank;
use  crate::types::morton::{KeyType, MortonKey};


pub const K: Rank = 2;

pub const NCRIT: usize = 150;

pub const DEEPEST_LEVEL: KeyType = 16;

pub const LEVEL_SIZE: KeyType = 1 << DEEPEST_LEVEL;

pub const ROOT: MortonKey = MortonKey{anchor: [0, 0, 0], morton: 0};