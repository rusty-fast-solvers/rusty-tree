//! Functions for data io and manipulation.
use vtkio::model::*;

use std::path::PathBuf;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;

use crate::types::{
    morton::MortonKey, point::Point
};

use serde::{Serialize, Deserialize};


// VTK compatible dataset for visualization
pub trait VTK {
    // Convert a data set to VTK format.
    fn to_vtk(&self, filename: String);
}

// impl VTK for Vec<MortonKey> {
//     fn to_vtk(&self, filename: String){

//         let data = Vec::<i32>::new();

//         let model = Vtk {
//             title: String::new(),
//             version: Version { major: 1, minor: 0 },
//             file_path:Some(PathBuf::from(&filename)), 
//             byte_order: ByteOrder::BigEndian,
//             data: DataSet::inline(UnstructuredGridPiece {
//                 points: None,
//                 cells: None,
//                 data: None,
//             }),
//         };
    
//         model.export(filename).unwrap();
//     }
// }

// impl VTK for Vec<Point> {
//     fn to_vtk(&self, filename: String){
        
//         let model = Vtk {
//             title: String::new(),
//             version: Version { major: 1, minor: 0 },
//             file_path:Some(PathBuf::from(&filename)), 
//             byte_order: ByteOrder::BigEndian,
//             data: DataSet::inline(UnstructuredGridPiece {
//                 points: None,
//                 cells: None,
//                 data: None,
//             }),
//         };
    
//         model.export(filename).unwrap();
//     }
// }

// Data input and output
pub trait JSON {

    // Save data to disk in JSON.
    fn write(&self, filename: String) -> Result<(), std::io::Error> 
        where Self: Serialize {

        let filepath = Path::new(&filename);
        let file = File::create(filepath)?;
        let writer = BufWriter::new(file);
        let result = serde_json::to_writer(writer, self)?;
        
        Ok(result)
    }

    fn read<'de, P: AsRef<Path>, T: serde::de::DeserializeOwned>(filepath: P) -> Result<Vec<T>, std::io::Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let result: Vec<T> = serde_json::from_reader(reader)?;
        Ok(result)
    }
}
