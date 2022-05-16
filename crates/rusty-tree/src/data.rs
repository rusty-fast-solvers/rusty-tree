//! Functions for data io and manipulation.
use vtkio::model::*;
use std::path::PathBuf;

use crate::types::{
    morton::MortonKey, point::Point
};

pub trait VTK {
    fn to_vtk(&self, filename: String){}
}

impl VTK for Vec<MortonKey> {
    fn to_vtk(&self, filename: String){


    }
}

impl VTK for Vec<Point> {
    fn to_vtk(&self, filename: String){
    }
}

