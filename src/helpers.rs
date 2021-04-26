//! This module contains useful helper functions.

use rusty_kernel_tools::RealType;
use ndarray::{Array1, ArrayView2, Axis};
use num;

/// Compute the center of a (3, N) array.
///
/// # Arguments
/// 
/// * `arr` - A (3, N) array.
pub fn compute_center<T: RealType>(arr: &ArrayView2<T>) -> Array1<T> {
    let mut result = arr.sum_axis(Axis(1));
    let len: T = num::traits::cast(arr.len_of(Axis(1))).unwrap();
    result.iter_mut().for_each(|elem| *elem = *elem / len);
    result
}

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
pub fn compute_bounds<T: RealType>(arr: &ArrayView2<T>) -> [[T; 2]; 3] {
    let mut bounds: [[T; 2]; 3] = [[num::traits::zero(); 2]; 3];

    arr.axis_iter(Axis(0)).enumerate().for_each(|(i, axis)| {
        let mut lower = T::infinity();
        let mut upper = T::neg_infinity();

        axis.iter().for_each(|val| {
            upper = val.max(upper);
            lower = val.min(lower)
        });

        bounds[i][0] = lower;
        bounds[i][1] = upper;
    });
    bounds
}
