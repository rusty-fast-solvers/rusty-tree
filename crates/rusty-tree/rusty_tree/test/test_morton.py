"""Test Morton keys."""
from rusty_tree.morton import MortonKey
from rusty_tree import LEVEL_DISPLACEMENT, DEEPEST_LEVEL

import numpy as np


def get_bit(number, index):
    """
    Get the bit of position `index` from `number`.
    """
    return (number >> index) & 1


def count_trailing_zero_bits(number):
    """Count the number of trailing zeros of a number."""

    count = 0

    while get_bit(number, count) == 0:
        count += 1
        if count == 64:
            break
    return count


def test_encode_point():
    """Test the encoding of a point."""

    origin = np.array([0.0, 0, 0])
    diameter = np.array([1.1, 2.0, 3.0])

    point = np.random.rand(3)

    key = MortonKey.from_point(point, origin, diameter)

    anchor_coord = key.to_coordinates(origin, diameter)

    box_diam = diameter / (1 << DEEPEST_LEVEL)

    assert np.all(point >= anchor_coord)
    assert np.all(point - anchor_coord < box_diam)


def test_parent():
    """Compute the hierarchy of parents and check their keys."""

    key = MortonKey.from_anchor(
        [(1 << DEEPEST_LEVEL) - 1, (1 << DEEPEST_LEVEL) - 1, (1 << DEEPEST_LEVEL) - 1]
    )
    morton = key.morton >> LEVEL_DISPLACEMENT
    expected_level = DEEPEST_LEVEL

    while key.level > 1:
        key = key.parent()
        expected_level -= 1
        level_diff = DEEPEST_LEVEL - expected_level
        expected_trailing_zeros = 3 * level_diff
        expected_morton = (morton >> expected_trailing_zeros) << expected_trailing_zeros
        actual_trailing_zeros = count_trailing_zero_bits(
            key.morton >> LEVEL_DISPLACEMENT
        )

        assert expected_trailing_zeros == actual_trailing_zeros
        assert expected_level == key.level
        assert expected_morton == key.morton >> LEVEL_DISPLACEMENT


def test_key_along_direction():
    """Test computing the key along a given direction."""

    key = MortonKey.from_anchor([1, 2, 3])

    new_key = key.find_key_in_direction([1, 5, 2])
    assert (new_key.anchor == [2, 7, 5]).all()
    new_key = key.find_key_in_direction([0, -1, 0])
    assert (new_key.anchor == [1, 1, 3]).all()
    new_key = key.find_key_in_direction([(1 << DEEPEST_LEVEL) - 1, 0, 0])
    assert new_key is None


def test_coordinates():
    """Test the coordinates of an anchor."""
    xind = 4098
    yind = 1355
    zind = 7269

    anchor = np.array([xind, yind, zind])

    origin = np.array([-5, 13, 9.1])
    diameter = np.array([13, 7, 2])

    step_size = diameter / (1 << DEEPEST_LEVEL)
    expected_coords = origin + anchor * step_size

    key = MortonKey.from_anchor(anchor)

    actual_coords = key.to_coordinates(origin, diameter)
    np.testing.assert_allclose(actual_coords, expected_coords)


def test_box_coordinates():
    """Test whether serialized box coordinates are correct."""
    xind = 4098
    yind = 1355
    zind = 7269

    anchor = np.array([xind, yind, zind])

    origin = np.array([-5, 13, 9.1])
    diameter = np.array([13, 7, 2])

    step_size = diameter / (1 << DEEPEST_LEVEL)
    mask = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    key = MortonKey.from_anchor(anchor)
    expected_coords = key.to_coordinates(origin, diameter) + mask * step_size

    actual_coords = key.box_coordinates(origin, diameter)

    np.testing.assert_allclose(actual_coords, expected_coords)


def test_ancestor_descendent():
    """Test if ancestor and descendent are correctly returned."""

    anchor = [
        (1 << DEEPEST_LEVEL) - 1,
        (1 << DEEPEST_LEVEL) - 1,
        (1 << DEEPEST_LEVEL) - 1,
    ]

    key = MortonKey.from_anchor(anchor)
    parent = key.parent()

    assert parent.is_ancestor(key)
    assert key.is_descendent(parent)
    assert key.is_ancestor(parent) is False
