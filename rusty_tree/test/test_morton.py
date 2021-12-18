"""Test Morton keys."""
from rusty_tree.morton import MortonKey
from rusty_tree import LEVEL_DISPLACEMENT, DEEPEST_LEVEL

import numpy as np

def get_bit(number, index):
    """
    Get the bit of position `index` from `number`.
    """
    return (number >> index) & 1

def count_trailing_zero_bits(morton):
    """Count the number of trailing zeros of a Morton key."""

    count = 0

    while get_bit(morton, count + LEVEL_DISPLACEMENT) == 0:
        count += 1
        if count == 64:
            break
    return count


def test_parent():
    """Test that the correct parent is returned."""

    key = MortonKey.from_anchor([1, 2, 3])
    parent = key.parent()

    expected_morton = ((key.morton >> LEVEL_DISPLACEMENT) >> 3) << 3
    actual_morton = parent.morton >> LEVEL_DISPLACEMENT

    assert expected_morton == actual_morton
    assert key.level - 1 == parent.level

def test_encode_point():
    """Test the encoding of a point."""

    origin = np.array([0., 0, 0])
    diameter = np.array([1.1, 2.0, 3.0])

    point = np.random.rand(3)

    key = MortonKey.from_point(point, origin, diameter)

    anchor_coord = key.to_coordinates(origin, diameter)

    box_diam = diameter / (1 << DEEPEST_LEVEL)

    assert np.all(point >= anchor_coord)
    assert np.all(point - anchor_coord < box_diam)

def test_parent():
    """Test computing the parent of a Morton key."""

    key = MortonKey.from_anchor([65535, 65535, 65535])

        

        morton = morton >> LEVEL_DISPLACEMENT


    # Have generated key on deepest level. Now iteratively call
    # parent routine and check that the right number of bits are zero
    # at the end.


