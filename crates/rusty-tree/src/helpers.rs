//! Assorted helper functions.

use crate::{
    types::{
        point::Point,
        domain::Domain
    },
};


/// Compute the points bounds for points on a local node
pub fn compute_bounds(points: &Vec<Point>) {

    let max_x = points.iter().map(|p| p.coordinate[0]); 
    // let max_y = points.iter().map(|p| y.coordinate[1]).collect().max().unwrap(); 
    // let max_z = points.iter().map(|p| z.coordinate[2]).collect().max().unwrap(); 
    
    // let min_x = points.iter().map(|p| p.coordinate[0]).collect().min().unwrap(); 
    // let min_y = points.iter().map(|p| y.coordinate[1]).collect().min().unwrap(); 
    // let min_z = points.iter().map(|p| z.coordinate[2]).collect().min().unwrap(); 

    println!("max {:?}", max_x);
}



/// Compute the points bounds over all nodes.
pub fn compute_bounds_global() {

}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_compute_bounds() {
        assert!(false);
    }
}

