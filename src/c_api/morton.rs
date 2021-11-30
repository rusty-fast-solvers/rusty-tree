// C-API for the Morton Key Type

use crate::morton::MortonKey;
use crate::types::{Domain, KeyType, PointType};
use std::mem;

#[no_mangle]
pub extern "C" fn morton_key_from_anchor(p_anchor: *const [KeyType; 3]) -> *mut MortonKey {
    let anchor: &[KeyType; 3] = unsafe { p_anchor.as_ref().unwrap() };

    get_raw(MortonKey::from_anchor(anchor))
}

#[no_mangle]
pub extern "C" fn morton_key_from_morton(morton: KeyType) -> *mut MortonKey {
    get_raw(MortonKey::from_morton(morton))
}

#[no_mangle]
pub extern "C" fn morton_key_from_point(
    p_point: *const [PointType; 3],
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
) -> *mut MortonKey {
    let point: &[PointType; 3] = unsafe { p_point.as_ref().unwrap() };
    let origin: &[PointType; 3] = unsafe { p_origin.as_ref().unwrap() };
    let diameter: &[PointType; 3] = unsafe { p_diameter.as_ref().unwrap() };

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };
    get_raw(MortonKey::from_point(point, &domain))
}

#[no_mangle]
pub extern "C" fn morton_key_parent(p_morton: *mut MortonKey) -> *mut MortonKey {
    let parent = unsafe { (*p_morton).parent() };
    get_raw(parent)
}

#[no_mangle]
pub extern "C" fn morton_key_first_child(p_morton: *mut MortonKey) -> *mut MortonKey {
    let first_child = unsafe { (*p_morton).first_child() };
    get_raw(first_child)
}

#[no_mangle]
pub extern "C" fn morton_key_children(p_morton: *mut MortonKey) -> *mut usize {
    let mut children_vec = unsafe { (*p_morton).children() };

    let mut children_boxes: Vec<usize> = vec![0; 8];
    children_boxes.shrink_to_fit();
    for index in 0..8 {
        let child = children_vec.pop().unwrap();
        children_boxes[7 - index] = get_raw(child) as usize;
    }

    let ptr = children_boxes.as_mut_ptr();

    mem::forget(children_boxes);

    ptr
}

#[no_mangle]
pub extern "C" fn morton_key_delete(p_morton_key: *mut MortonKey) {
    unsafe {
        drop(Box::from_raw(p_morton_key));
    }
}

#[no_mangle]
pub extern "C" fn delete_usize_vec(ptr: *mut usize, length: usize) {
    let vec = unsafe { Vec::from_raw_parts(ptr, length, length) };
    drop(vec)
}

/// Return a raw pointer for an object
fn get_raw(key: MortonKey) -> *mut MortonKey {
    Box::into_raw(Box::new(key))
}
