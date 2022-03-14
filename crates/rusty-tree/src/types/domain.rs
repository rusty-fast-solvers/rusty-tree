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

pub struct Domain {
    pub origin: [PointType; 3],
    pub diameter: [PointType; 3],
}

#[derive(Debug, Clone)]
pub struct MortonDomain {
    pub rank: Rank,
    pub left: MortonKey,
    pub right: MortonKey,
}

unsafe impl Equivalence for MortonDomain {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                offset_of!(MortonDomain, rank) as Address,
                offset_of!(MortonDomain, left) as Address,
                offset_of!(MortonDomain, right) as Address
            ],
            &[
                UncommittedUserDatatype::contiguous(1, &Rank::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::structured(
                    &[1, 1],
                    &[
                        offset_of!(MortonKey, anchor) as Address,
                        offset_of!(MortonKey, morton) as Address
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(3, &KeyType::equivalent_datatype()).as_ref(),
                        UncommittedUserDatatype::contiguous(1, &KeyType::equivalent_datatype()).as_ref(),
                    ]
                ).as_ref(),
                UncommittedUserDatatype::structured(
                    &[1, 1],
                    &[
                        offset_of!(MortonKey, anchor) as Address,
                        offset_of!(MortonKey, morton) as Address
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(3, &KeyType::equivalent_datatype()).as_ref(),
                        UncommittedUserDatatype::contiguous(1, &KeyType::equivalent_datatype()).as_ref(),
                    ]
                ).as_ref() 
            ]
        )
    }
}