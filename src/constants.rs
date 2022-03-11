//! Crate wide constants
use  crate::types::morton::{KeyType, MortonKey};

pub const DEEPEST_LEVEL: KeyType = 16;
pub const LEVEL_SIZE: KeyType = 1 << DEEPEST_LEVEL;
pub const ROOT: MortonKey = MortonKey{anchor: [0, 0, 0], morton: 0};