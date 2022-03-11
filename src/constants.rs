//! Crate wide constants
use  morton::{KeyType, Key};

pub const DEEPEST_LEVEL: KeyType = 16;
pub const LEVEL_SIZE: KeyType = 1 << DEEPEST_LEVEL;
pub const ROOT: Key = Key{anchor: [0, 0, 0], morton: 0};