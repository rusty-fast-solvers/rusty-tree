// C-API for the Morton Key Type

use crate::types::{
    domain::Domain,
    morton::{MortonKey, KeyType},
    point::PointType
};


#[no_mangle]
pub extern "C" fn morton_key_from_anchor(p_anchor: *const [KeyType; 3]) -> *mut MortonKey {
    let anchor: &[KeyType; 3] = unsafe { p_anchor.as_ref().unwrap() };
    Box::into_raw(Box::new(MortonKey::from_anchor(anchor)))
}

#[no_mangle]
pub extern "C" fn morton_key_from_morton(morton: KeyType) -> *mut MortonKey {
    Box::into_raw(Box::new(MortonKey::from_morton(morton)))
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

    Box::into_raw(Box::new(MortonKey::from_point(point, &domain)))
}

#[no_mangle]
pub extern "C" fn morton_key_parent(p_morton: *mut MortonKey) -> *mut MortonKey {
    let parent = unsafe { (*p_morton).parent() };
    Box::into_raw(Box::new(parent))
}

#[no_mangle]
pub extern "C" fn morton_key_level(p_morton: *mut MortonKey) -> KeyType {
    unsafe { (*p_morton).level() }
}

#[no_mangle]
pub extern "C" fn morton_key_first_child(p_morton: *mut MortonKey) -> *mut MortonKey {
    let first_child = unsafe { (*p_morton).first_child() };
    Box::into_raw(Box::new(first_child))
}

#[no_mangle]
pub extern "C" fn morton_key_children(p_morton: *mut MortonKey, ptr: *mut usize) {
    let mut children_vec = unsafe { (*p_morton).children() };

    let children_boxes = unsafe { std::slice::from_raw_parts_mut(ptr, 8) };
    for index in 0..8 {
        let child = children_vec.pop().unwrap();
        children_boxes[7 - index] = Box::into_raw(Box::new(child)) as usize;
    }
}

#[no_mangle]
pub extern "C" fn morton_key_to_coordinates(
    p_morton: *mut MortonKey,
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
    p_coord: *mut [PointType; 3],
) {
    let origin: &[PointType; 3] = unsafe { p_origin.as_ref().unwrap() };
    let diameter: &[PointType; 3] = unsafe { p_diameter.as_ref().unwrap() };

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };

    let tmp = unsafe { (*p_morton).to_coordinates(&domain) };

    unsafe {
        for index in 0..3 {
            (*p_coord)[index] = tmp[index]
        }
    }
}

#[no_mangle]
pub extern "C" fn morton_key_box_coordinates(
    p_morton: *mut MortonKey,
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
    box_coord: *mut [PointType; 24],
) {
    let origin: &[PointType; 3] = unsafe { p_origin.as_ref().unwrap() };
    let diameter: &[PointType; 3] = unsafe { p_diameter.as_ref().unwrap() };

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };

    let coords = unsafe { (*p_morton).box_coordinates(&domain) };

    unsafe {
        for index in 0..24 {
            (*box_coord)[index] = coords[index];
        }
    }
}

#[no_mangle]
pub extern "C" fn morton_key_key_in_direction(
    p_morton: *mut MortonKey,
    p_direction: *const [i64; 3],
) -> *mut MortonKey {
    let direction = unsafe { p_direction.as_ref().unwrap() };

    let shifted_key = unsafe { (*p_morton).find_key_in_direction(direction) };

    match shifted_key {
        Some(key) =>
            Box::into_raw(Box::new(key)),

        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn morton_key_is_ancestor(
    p_morton: *mut MortonKey,
    p_other: *mut MortonKey,
) -> bool {
    unsafe { (*p_morton).is_ancestor(&*p_other)}
}

#[no_mangle]
pub extern "C" fn morton_key_is_descendent(
    p_morton: *mut MortonKey,
    p_other: *mut MortonKey,
) -> bool {
    unsafe { (*p_morton).is_descendent(&*p_other)}
}


#[no_mangle]
pub extern "C" fn morton_key_delete(p_morton_key: *mut MortonKey) {
    unsafe {
        drop(Box::from_raw(p_morton_key));
    }
}

#[no_mangle]
pub extern "C" fn morton_key_next(ptr: *const MortonKey) -> *mut &'static MortonKey {
    let mut slice = unsafe {std::slice::from_raw_parts(ptr, 2).iter()};
    slice.next();
    let next = slice.next().unwrap();
    Box::into_raw(Box::new(next))
}


#[no_mangle]
pub extern "C" fn morton_key_slice(
    p_keys: *const MortonKey,
    ptr: *mut usize,
    nkeys: usize,
    lidx: usize,
    ridx: usize
) {
    let mut keys = unsafe {std::slice::from_raw_parts(p_keys, nkeys).iter()};
    let mut i = 0;

    while i < lidx {
        keys.next();
        i += 1;
    }

    let nslice = ridx-lidx;
    
    let boxes = unsafe {std::slice::from_raw_parts_mut(ptr, nslice)};

    for i in 0..nslice {
        let key = keys.next().unwrap().clone();
        boxes[i] = Box::into_raw(Box::new(key)) as usize;
    }
}
