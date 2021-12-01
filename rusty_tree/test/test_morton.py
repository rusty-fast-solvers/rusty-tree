"""Test Morton keys."""
from rusty_tree.morton import MortonKey
from rusty_tree import LEVEL_DISPLACEMENT, DEEPEST_LEVEL

import numpy as np


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


