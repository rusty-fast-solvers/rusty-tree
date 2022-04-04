use crate::types::point::Point;


#[no_mangle]
pub extern "C" fn point_next(ptr: *const Point) -> *mut &'static Point {
    let mut slice = unsafe {std::slice::from_raw_parts(ptr, 2).iter()};
    slice.next();
    let next = slice.next().unwrap();
    Box::into_raw(Box::new(next))
}

#[no_mangle]
pub extern "C" fn point_slice(
    p_points: *const Point,
    ptr: *mut usize,
    npoints: usize,
    lidx: usize,
    ridx: usize
) {
    let mut points = unsafe {std::slice::from_raw_parts(p_points, npoints).iter()};
    let mut i = 0;

    while i < lidx {
        points.next();
        i += 1;
    }

    let nslice = ridx-lidx;
    
    let boxes = unsafe {std::slice::from_raw_parts_mut(ptr, nslice)};

    for i in 0..nslice {
        let point = points.next().unwrap().clone();
        boxes[i] = Box::into_raw(Box::new(point)) as usize;
    }
}
