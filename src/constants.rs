//! Crate-wide constants.

use crate::types::KeyType;

// Number of bits used for Level information.
pub const LEVEL_DISPLACEMENT: usize = 15;

// Mask for the last 15 bits.
pub const LEVEL_MASK: KeyType = 0x7FFF;

// Mask for lowest order byte.
pub const BYTE_MASK: KeyType = 0xFF;
pub const BYTE_DISPLACEMENT: KeyType = 8;

// Mask encapsulating a bit.
pub const NINE_BIT_MASK: KeyType = 0x1FF;

pub const DEEPEST_LEVEL: KeyType = 16;
pub const LEVEL_SIZE: KeyType = 1 << DEEPEST_LEVEL;