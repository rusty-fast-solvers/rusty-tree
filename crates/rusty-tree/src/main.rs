use rusty_tree::types::point::Point;
use rusty_tree::types::morton::MortonKey;
use rusty_tree::data::{HDF5};

fn main() {

    let key = MortonKey {anchor: [0, 0, 0], morton: 0};

    let points = vec![
        Point {
            global_idx: 0,
            coordinate: [0.0, 0.0, 0.0],
            key: key
        }
    ];

    // Write hdf5
    // points.write_hdf5("test.h5");

    // Read hdf5
    // let read: Vec<Point> = Vec::<Point>::read_hdf5("test.h5").unwrap();
    // println!("here {:?}", points);
    // println!("here {:?}", read);
}