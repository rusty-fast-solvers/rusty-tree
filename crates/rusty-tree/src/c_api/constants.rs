// C API for project constants

use crate::types::morton::KeyType;
use crate::constants;

#[no_mangle]
pub static DIRECTIONS: [[i64; 3]; 26] = constants::DIRECTIONS;

#[no_mangle]
pub static Z_LOOKUP_ENCODE: [KeyType; 256] = constants::Z_LOOKUP_ENCODE;

#[no_mangle]
pub static Y_LOOKUP_ENCODE: [KeyType; 256] = constants::Y_LOOKUP_ENCODE;

#[no_mangle]
pub static X_LOOKUP_ENCODE: [KeyType; 256] = constants::X_LOOKUP_ENCODE;

#[no_mangle]
pub static Z_LOOKUP_DECODE: [KeyType; 512] = constants::Z_LOOKUP_DECODE;

#[no_mangle]
pub static Y_LOOKUP_DECODE: [KeyType; 512] = constants::Y_LOOKUP_DECODE;

#[no_mangle]
pub static X_LOOKUP_DECODE: [KeyType; 512] = constants::X_LOOKUP_DECODE;