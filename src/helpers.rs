//! This module contains useful helper functions.

use ndarray::{ArrayView2, Axis};
use num;
use rusty_kernel_tools::RealType;


/// Compute the bounds of a (3, N) array
///
/// This function returns a (3, 2) array `bounds`, where `bounds[i][0]`
/// is the lower bound along the ith axis, and `bounds[i][1]` is the
/// upper bound along the jth axis.
///
/// # Arguments
///
/// * `arr` - A (3, N) array.
///
pub fn compute_bounds<T: RealType>(arr: ArrayView2<T>) -> [[T; 2]; 3] {
    let mut bounds: [[T; 2]; 3] = [[num::traits::zero(); 2]; 3];

    arr.axis_iter(Axis(0)).enumerate().for_each(|(i, axis)| {
        bounds[i][0] = axis.iter().copied().reduce(T::min).unwrap();
        bounds[i][1] = axis.iter().copied().reduce(T::max).unwrap()
    });

    bounds
}

