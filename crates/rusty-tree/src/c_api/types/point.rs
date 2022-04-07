use crate::types::point::Point;


#[no_mangle]
pub extern "C" fn point_next(ptr: *const Point) -> *mut &'static Point {
    let mut slice = unsafe {std::slice::from_raw_parts(ptr, 2).iter()};
    slice.next();
    let next = slice.next().unwrap();
    Box::into_raw(Box::new(next))
}

#[no_mangle]
pub extern "C" fn point_clone(
    ptr: *const Point,
    data_ptr: *mut usize,
    len: usize,
    start: usize,
    stop: usize
) {
    let slice = unsafe {std::slice::from_raw_parts(ptr, len)};

    let nslice = stop-start;
    let boxes = unsafe {std::slice::from_raw_parts_mut(data_ptr, nslice)};

    let mut jdx = 0;
    for idx in start..stop {
        boxes[jdx] = Box::into_raw(Box::new(slice[idx])) as usize;
        jdx += 1;
    }
}

#[no_mangle]
pub extern "C" fn point_index(ptr: *const Point, len: usize, idx: usize) -> *mut &'static Point {
    let slice = unsafe {std::slice::from_raw_parts(ptr, len)};
    Box::into_raw(Box::new(&slice[idx]))
}
