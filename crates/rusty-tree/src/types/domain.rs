//! Definition of basic types

use memoffset::offset_of;
use mpi::{
    Address,
    datatype::{
        Equivalence, UncommittedUserDatatype, UserDatatype
    },
    topology::Rank
};

use crate::{
    types::{
        morton::{MortonKey, KeyType},
        point::PointType
    }
};


#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub origin: [PointType; 3],
    pub diameter: [PointType; 3],
}

unsafe impl Equivalence for Domain {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(Domain, origin) as Address,
                offset_of!(Domain, diameter) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &PointType::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(3, &PointType::equivalent_datatype()).as_ref(),
            ]
        )
    }
}
