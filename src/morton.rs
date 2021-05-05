//! Routines for Morton encoding and decoding.

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};
use rusty_kernel_tools::RealType;
use std::collections::{HashSet};

const X_LOOKUP_ENCODE: [usize; 256] = [
    0x00000000, 0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049,
    0x00000200, 0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249,
    0x00001000, 0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049,
    0x00001200, 0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249,
    0x00008000, 0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049,
    0x00008200, 0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249,
    0x00009000, 0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049,
    0x00009200, 0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249,
    0x00040000, 0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049,
    0x00040200, 0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249,
    0x00041000, 0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049,
    0x00041200, 0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249,
    0x00048000, 0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049,
    0x00048200, 0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249,
    0x00049000, 0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049,
    0x00049200, 0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249,
    0x00200000, 0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049,
    0x00200200, 0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249,
    0x00201000, 0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049,
    0x00201200, 0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249,
    0x00208000, 0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049,
    0x00208200, 0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249,
    0x00209000, 0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049,
    0x00209200, 0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249,
    0x00240000, 0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049,
    0x00240200, 0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249,
    0x00241000, 0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049,
    0x00241200, 0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249,
    0x00248000, 0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049,
    0x00248200, 0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249,
    0x00249000, 0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049,
    0x00249200, 0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249,
];

const Y_LOOKUP_ENCODE: [usize; 256] = [
    0x00000000, 0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092,
    0x00000400, 0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492,
    0x00002000, 0x00002002, 0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092,
    0x00002400, 0x00002402, 0x00002410, 0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492,
    0x00010000, 0x00010002, 0x00010010, 0x00010012, 0x00010080, 0x00010082, 0x00010090, 0x00010092,
    0x00010400, 0x00010402, 0x00010410, 0x00010412, 0x00010480, 0x00010482, 0x00010490, 0x00010492,
    0x00012000, 0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082, 0x00012090, 0x00012092,
    0x00012400, 0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490, 0x00012492,
    0x00080000, 0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092,
    0x00080400, 0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492,
    0x00082000, 0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092,
    0x00082400, 0x00082402, 0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492,
    0x00090000, 0x00090002, 0x00090010, 0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092,
    0x00090400, 0x00090402, 0x00090410, 0x00090412, 0x00090480, 0x00090482, 0x00090490, 0x00090492,
    0x00092000, 0x00092002, 0x00092010, 0x00092012, 0x00092080, 0x00092082, 0x00092090, 0x00092092,
    0x00092400, 0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482, 0x00092490, 0x00092492,
    0x00400000, 0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090, 0x00400092,
    0x00400400, 0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492,
    0x00402000, 0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092,
    0x00402400, 0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492,
    0x00410000, 0x00410002, 0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092,
    0x00410400, 0x00410402, 0x00410410, 0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492,
    0x00412000, 0x00412002, 0x00412010, 0x00412012, 0x00412080, 0x00412082, 0x00412090, 0x00412092,
    0x00412400, 0x00412402, 0x00412410, 0x00412412, 0x00412480, 0x00412482, 0x00412490, 0x00412492,
    0x00480000, 0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082, 0x00480090, 0x00480092,
    0x00480400, 0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490, 0x00480492,
    0x00482000, 0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092,
    0x00482400, 0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492,
    0x00490000, 0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092,
    0x00490400, 0x00490402, 0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492,
    0x00492000, 0x00492002, 0x00492010, 0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092,
    0x00492400, 0x00492402, 0x00492410, 0x00492412, 0x00492480, 0x00492482, 0x00492490, 0x00492492,
];

const Z_LOOKUP_ENCODE: [usize; 256] = [
    0x00000000, 0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124,
    0x00000800, 0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924,
    0x00004000, 0x00004004, 0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124,
    0x00004800, 0x00004804, 0x00004820, 0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924,
    0x00020000, 0x00020004, 0x00020020, 0x00020024, 0x00020100, 0x00020104, 0x00020120, 0x00020124,
    0x00020800, 0x00020804, 0x00020820, 0x00020824, 0x00020900, 0x00020904, 0x00020920, 0x00020924,
    0x00024000, 0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104, 0x00024120, 0x00024124,
    0x00024800, 0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920, 0x00024924,
    0x00100000, 0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124,
    0x00100800, 0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924,
    0x00104000, 0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124,
    0x00104800, 0x00104804, 0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924,
    0x00120000, 0x00120004, 0x00120020, 0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124,
    0x00120800, 0x00120804, 0x00120820, 0x00120824, 0x00120900, 0x00120904, 0x00120920, 0x00120924,
    0x00124000, 0x00124004, 0x00124020, 0x00124024, 0x00124100, 0x00124104, 0x00124120, 0x00124124,
    0x00124800, 0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904, 0x00124920, 0x00124924,
    0x00800000, 0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120, 0x00800124,
    0x00800800, 0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924,
    0x00804000, 0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124,
    0x00804800, 0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924,
    0x00820000, 0x00820004, 0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124,
    0x00820800, 0x00820804, 0x00820820, 0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924,
    0x00824000, 0x00824004, 0x00824020, 0x00824024, 0x00824100, 0x00824104, 0x00824120, 0x00824124,
    0x00824800, 0x00824804, 0x00824820, 0x00824824, 0x00824900, 0x00824904, 0x00824920, 0x00824924,
    0x00900000, 0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104, 0x00900120, 0x00900124,
    0x00900800, 0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920, 0x00900924,
    0x00904000, 0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124,
    0x00904800, 0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924,
    0x00920000, 0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124,
    0x00920800, 0x00920804, 0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924,
    0x00924000, 0x00924004, 0x00924020, 0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124,
    0x00924800, 0x00924804, 0x00924820, 0x00924824, 0x00924900, 0x00924904, 0x00924920, 0x00924924,
];

const X_LOOKUP_DECODE: [usize; 512] = [
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
];

const Y_LOOKUP_DECODE: [usize; 512] = [
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
    4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
];

const Z_LOOKUP_DECODE: [usize; 512] = [
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
];

// Number of bits used for Level information
const LEVEL_DISPLACEMENT: usize = 15;

// Mask for the last 15 bits
const LEVEL_MASK: usize = 0x7FFF;

// Mask for lowest order byte
const BYTE_MASK: usize = 0xFF;
const BYTE_DISPLACEMENT: usize = 8;

// Mask encapsulating a bit
const NINE_BIT_MASK: usize = 0x1FF;

/// Return the level associated with a key.
pub fn find_level(key: usize) -> usize {
    return key & LEVEL_MASK;
}

/// Map a point to the integer coordinates of its enclosing box.
///
/// Returns the 3 integeger coordinates of the enclosing box.
///
/// # Arguments
/// `point` - The (x, y, z) coordinates of the point to map.
/// `level` - The level of the tree at which the point will be mapped.
/// `tree_center` - The center of the octree.
/// `tree_radius` - The radius of the octree.
pub fn point_to_box_coordinates(
    point: &[f64; 3],
    level: usize,
    tree_center: &[f64; 3],
    tree_radius: f64,
) -> [usize; 4] {
    use itertools::izip;
    let mut anchor: [usize; 4] = [0, 0, 0, 0];
    anchor[3] = level;

    let side_length = 2.0 * tree_radius / ((1 << level) as f64);

    for (anchor_value, point_value, tree_center_value) in izip!(&mut anchor, point, tree_center) {
        *anchor_value =
            ((point_value - (tree_center_value - tree_radius)) / side_length).floor() as usize;
    }

    anchor
}

/// Encode an anchor.
///
/// Returns the Morton key associated with the given anchor.
///
/// # Arguments
/// `anchor` - A vector with 4 elements defining the integer coordinates and level.
pub fn encode_anchor(anchor: &[usize; 4]) -> usize {
    let x = anchor[0];
    let y = anchor[1];
    let z = anchor[2];
    let level = anchor[3];

    let key: usize = Z_LOOKUP_ENCODE[(z >> BYTE_DISPLACEMENT) & BYTE_MASK]
        | Y_LOOKUP_ENCODE[(y >> BYTE_DISPLACEMENT) & BYTE_MASK]
        | X_LOOKUP_ENCODE[(x >> BYTE_DISPLACEMENT) & BYTE_MASK];

    let key = (key << 24)
        | Z_LOOKUP_ENCODE[z & BYTE_MASK]
        | Y_LOOKUP_ENCODE[y & BYTE_MASK]
        | X_LOOKUP_ENCODE[x & BYTE_MASK];

    let key = key << LEVEL_DISPLACEMENT;
    key | level
}

/// Encode a point.
///
/// Return the Morton key of a point for a given level.
///
/// # Arguments
/// `point` - The (x, y, z) coordinates of the point to map.
/// `level` - The level of the tree at which the point will be mapped.
/// `tree_center` - The center of the octree.
/// `tree_radius` - The radius of the octree.
pub fn encode_point(
    point: &[f64; 3],
    level: usize,
    tree_center: &[f64; 3],
    tree_radius: f64,
) -> usize {
    let anchor = point_to_box_coordinates(point, level, &tree_center, tree_radius);
    encode_anchor(&anchor)
}

/// Given an anchor, return the corresponding x,y,z coordinates.
///
/// The result is the coordinates of the lower left corner of the box described by the given anchor.
///
/// # Arguments
/// `anchor` - The three indices describing the box and the level information.
/// `tree_center - The center of the tree.
/// `tree_radius` - The tree radius.
pub fn anchor_to_coordinates(
    anchor: &[usize; 4],
    tree_center: &[f64; 3],
    tree_radius: f64,
) -> [f64; 3] {
    use itertools::izip;
    let mut coord: [f64; 3] = [0.0; 3];

    let level = anchor[3];
    let side_length = 2.0 * tree_radius / ((1 << level) as f64);

    for (&center, &anchor_value, coord_ref) in izip!(tree_center, anchor, &mut coord) {
        *coord_ref = (center - tree_radius) + side_length * (anchor_value as f64);
    }

    coord
}

/// Encode many points.
///
/// Return an array containing all Morton keys of a given array of points.
///
/// # Arguments
/// `point` - A (3 ,N) array of N points of type f32 or f64.
/// `level` - The level of the tree at which the point will be mapped.
/// `tree_center` - The center of the octree.
/// `tree_radius` - The radius of the octree.
pub fn encode_points<T: RealType>(
    points: ArrayView2<T>,
    level: usize,
    tree_center: &[f64; 3],
    tree_radius: f64,
) -> Array1<usize> {
    let npoints = points.len_of(Axis(1));
    let mut box_coordinates = Array2::<usize>::zeros((3, npoints));
    let mut keys = Array1::<usize>::zeros(npoints);

    let side_length = 2.0 * tree_radius / ((1 << level) as f64);
    let side_length = num::cast::cast::<f64, T>(side_length).unwrap();

    let mut min_bound = Array1::<T>::zeros(3);

    for (index, min_ref) in min_bound.iter_mut().enumerate() {
        *min_ref = num::cast::cast::<f64, T>(tree_center[index] - tree_radius).unwrap();
    }

    Zip::from(points.axis_iter(Axis(0)))
        .and(box_coordinates.axis_iter_mut(Axis(0)))
        .and(min_bound.view())
        .for_each(|points_row, box_coordinates_row, &min_value| {
            Zip::from(points_row).and(box_coordinates_row).par_for_each(
                |&point_value, box_coordinate_value| {
                    let tmp = (point_value - min_value) / side_length;
                    *box_coordinate_value = tmp.floor().to_usize().unwrap();
                },
            )
        });

    Zip::from(keys.view_mut())
        .and(box_coordinates.axis_iter(Axis(1)))
        .par_for_each(|key, box_coordinate| {
            #[test]
            fn test_y_encode_table() {
                for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
                    let mut sum: usize = 0;
                    for shift in 0..8 {
                        sum += (index & 1) << (3 * shift + 1);
                        index = index >> 1;
                    }
                    assert_eq!(sum, *actual);
                }
            }
            let anchor: [usize; 4] = [
                box_coordinate[0],
                box_coordinate[1],
                box_coordinate[2],
                level,
            ];
            *key = encode_anchor(&anchor);
        });

    keys
}

/// Helper function for decoding keys.
fn decode_key_helper(key: usize, lookup_table: &[usize; 512]) -> usize {
    const N_LOOPS: usize = 7; // 8 bytes in 64 bit key
    let mut coord: usize = 0;

    for index in 0..N_LOOPS {
        coord |= lookup_table[(key >> (index * 9)) & NINE_BIT_MASK] << (3 * index);
    }

    coord
}

/// Decode a given key.
///
/// Returns an array containing the three coordinates and level of the key.
pub fn decode_key(key: usize) -> [usize; 4] {
    let level = find_level(key);
    let key = key >> LEVEL_DISPLACEMENT;

    let x = decode_key_helper(key, &X_LOOKUP_DECODE);
    let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
    let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

    [x, y, z, level]
}

/// Return the key of the parent node.
pub fn find_parent(key: usize) -> usize {
    let level = find_level(key);
    let key = key >> LEVEL_DISPLACEMENT;

    let parent_level = level - 1;

    ((key >> 3) << LEVEL_DISPLACEMENT) | parent_level
}

/// Return all children of a given key.
pub fn find_children(key: usize) -> [usize; 8] {
    let level = find_level(key);
    let key = key >> LEVEL_DISPLACEMENT;

    let mut children: [usize; 8] = [0; 8];

    let root = (key >> 3) << 3;

    for (index, item) in children.iter_mut().enumerate() {
        *item = ((root | index) << LEVEL_DISPLACEMENT) | (level + 1);
    }

    children
}

/// Return all siblings of a key.
///
/// For a given key this function returns all 8 children
/// of the parent of the key. Hence, the key itself is
/// returned as well.
pub fn find_siblings(key: usize) -> [usize; 8] {
    let parent = find_parent(key);
    find_children(parent)
}

/// Find key in a given direction.
///
/// Returns the key obtained by moving direction[j] boxes into direction j
/// starting from the anchor associated with the given key.
/// Negative steps are possible. If the result is out of bounds,
/// i.e.. anchor[j] + direction[j is negative or larger than the number of boxes
/// across each dimension, `None` is returned. Otherwise, `Some(new_key)` is returned,
/// where `new_key` is the Morton key after moving into the given direction.
///
/// # Arguments
/// `key` - The starting key.
/// `direction` - A vector describing how many boxes we move along each coordinate direction.
///               Negative values are possible (meaning that we move backwards).
pub fn find_key_in_direction(key: usize, direction: &[i64; 3]) -> Option<usize> {
    let anchor = decode_key(key);

    let level = anchor[3];

    let max_number_of_boxes: i64 = 1 << level;

    let x: i64 = anchor[0] as i64;
    let y: i64 = anchor[1] as i64;
    let z: i64 = anchor[2] as i64;

    let x = x + direction[0];
    let y = y + direction[1];
    let z = z + direction[2];

    if (x >= 0)
        & (y >= 0)
        & (z >= 0)
        & (x < max_number_of_boxes)
        & (y < max_number_of_boxes)
        & (z < max_number_of_boxes)
    {
        let new_anchor: [usize; 4] = [x as usize, y as usize, z as usize, level];
        Some(encode_anchor(&new_anchor))
    } else {
        None
    }
}

/// Compute nearfield
///
/// The nearfield is the set of all boxes that are bordering the current box, including the box itself.
///
/// # Arguments
/// `key` - The key for which we want to compute the neighbours.
pub fn compute_nearfield(key: usize) -> HashSet<usize> {
    let mut near_field = HashSet::<usize>::new();

    use itertools::iproduct;

    for (i, j, k) in iproduct!(0..3, 0..3, 0..3) {
        let direction: [i64; 3] = [i - 1, j - 1, k - 1];
        if let Some(key) = find_key_in_direction(key, &direction) {
            near_field.insert(key);
        }
    }
    near_field
}

/// Compute interaction list
/// 
/// The interaction list of a key consists of all the children of the near field of the
/// parent that are not themselves in the near field of the key.
/// The function returns a set of all keys that form the interaction list of the
/// current key.
pub fn compute_interaction_list(key: usize) -> HashSet<usize> {

    let mut interaction_list = HashSet::<usize>::new();
    let near_field = compute_nearfield(key);

    let parent = find_parent(key);
    let parent_near_field = compute_nearfield(parent);

    for &parent_neighbour in parent_near_field.iter() {

        let children = find_children(parent_neighbour);
        for &child in children.iter() {
            if !near_field.contains(&child) {
                interaction_list.insert(child);
            }
        }

    }

    interaction_list

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the encoding table for the x-coordinate.
    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: usize = 0;

            for shift in 0..8 {
                sum |= (index & 1) << (3 * shift);
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    /// Test the encoding table for the y-coordinate.
    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: usize = 0;

            for shift in 0..8 {
                sum |= (index & 1) << (3 * shift + 1);
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    /// Test the encoding table for the z-coordinate.
    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: usize = 0;

            for shift in 0..8 {
                sum |= (index & 1) << (3 * shift + 2);
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    /// Test the decoding table for the x-coordinate.
    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: usize = index & 1;
            expected |= ((index >> 3) & 1) << 1;
            expected |= ((index >> 6) & 1) << 2;

            assert_eq!(actual, expected);
        }
    }

    /// Test the decoding table for the y-coordinate.
    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: usize = (index >> 1) & 1;
            expected |= ((index >> 4) & 1) << 1;
            expected |= ((index >> 7) & 1) << 2;

            assert_eq!(actual, expected);
        }
    }

    /// Test the decoding table for the z-coordinate.
    #[test]
    fn test_z_decode_table() {
        let mut expected: usize = 0;
        for (index, &actual) in Z_LOOKUP_DECODE.iter().enumerate() {
            expected = (index >> 2) & 1;
            expected |= ((index >> 5) & 1) << 1;
            expected |= ((index >> 8) & 1) << 2;

            assert_eq!(actual, expected);
        }
    }

    /// Test encoding and decoding an anchor
    #[test]
    fn test_encoding_decodiing() {
        let anchor: [usize; 4] = [65535, 65535, 65535, 16];

        let actual = decode_key(encode_anchor(&anchor));

        assert_eq!(anchor, actual);
    }

    /// Test encoding many points
    #[test]
    fn test_encode_many_points() {
        use rand::prelude::*;

        const NPOINTS: usize = 100;
        const LEVEL: usize = 4;

        let mut rng = rand::thread_rng();

        let mut points = Array2::<f64>::zeros((3, NPOINTS));

        points.iter_mut().for_each(|item| *item = rng.gen::<f64>());

        let tree_center = [0.65, 0.5, 0.4]; // Don't just choose the mid-point of the unit interval as centre

        let tree_radius = 0.65;
        let keys = encode_points(points.view(), LEVEL, &tree_center, tree_radius);

        for (point, &key) in points.axis_iter(Axis(1)).zip(keys.iter()) {
            let point_arr = [point[0], point[1], point[2]];
            let single_key = encode_point(&point_arr, LEVEL, &tree_center, tree_radius);

            // Check that the key via encode_point is the same as the key via encode_points

            assert_eq!(single_key, key);

            // Check if box is close to the point.

            let box_size = 2.0 * tree_radius / ((1 << LEVEL) as f64);

            let anchor = decode_key(single_key);
            let coords = anchor_to_coordinates(&anchor, &tree_center, tree_radius);
            for dim in 0..3 {
                assert!(coords[dim] <= point_arr[dim]);
                assert!(point_arr[dim] < coords[dim] + box_size);
            }
        }
    }
}
