//! Definition of basic types

pub type PointType = f64;

#[derive(Clone, Copy, Debug)]
/// **Point**, Cartesian coordinates (x, y, z).
pub struct Point {
    pub coord: [PointType; 3],
    pub global_idx: usize,
}

pub struct Domain {
    pub origin: [PointType; 3],
    pub diameter: [PointType; 3],
}

/// Vector of **Points**.
pub type Points = Vec<Point>;

pub type KeyType = u64;


// impl Default for Key {
//     fn default() -> Self {
//         Key {
//             value: [0, 0, 0],
//             level: 0,
//             morton: 0,
//         }
//     }
// }

// impl Default for Point {
//     fn default() -> Self {
//         Point {
//             coord: [PointType::NAN, PointType::NAN, PointType::NAN],
//             global_idx: 0,
//         }
//     }
// }

// /// Subroutine in less than function, equivalent to comparing floor of log_2(x). Adapted from [3].
// /// Returns true if y has the most significant bit, false otherwise.
// /// # Example:
// /// If x= 3 and y = 4 then the bit representations are x = 0b011 and y = 0b100.
// /// The function will return true.
// /// If x = 5 and y = 4 then the bit representations are x = 0b101 and y = 0b100.
// /// The function will return false.
// fn y_has_most_significant_bit(x: u64, y: u64) -> bool {
//     (x < y) & (x < (x ^ y))
// }

// /// Implementation of Algorithm 12 in [1]. to compare the ordering of two **Morton Keys**. If key
// /// `a` is less than key `b`, this function evaluates to true.
// fn less_than(a: &Key, b: &Key) -> Option<bool> {
//     // If anchors match, the one at the coarser level has the lesser Morton id.
//     let same_anchor = a.value == b.value;

//     match same_anchor {
//         true => {
//             if a.level < b.level {
//                 Some(true)
//             } else {
//                 Some(false)
//             }
//         }
//         false => {
//             let x: Vec<KeyType> = vec![
//                 a.value[0] ^ b.value[0],
//                 a.value[1] ^ b.value[1],
//                 a.value[2] ^ b.value[2],
//             ];

//             let mut argmax: usize = 0;

//             for dim in 1..3 {
//                 if y_has_most_significant_bit(x[argmax], x[dim]) {
//                     argmax = dim
//                 }
//             }

//             if a.value[argmax] < b.value[argmax] {
//                 Some(true)
//             } else {
//                 Some(false)
//             }
//         }
//     }
// }

// impl Ord for Key {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }

// impl PartialOrd for Key {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         let less = less_than(self, other).unwrap();
//         let eq = self.eq(other);

//         match eq {
//             true => Some(Ordering::Equal),
//             false => match less {
//                 true => Some(Ordering::Less),
//                 false => Some(Ordering::Greater),
//             },
//         }
//     }
// }
