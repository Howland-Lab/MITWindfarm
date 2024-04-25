import pytest
import numpy as np
from mitwindfarm.Layout import Layout


@pytest.fixture
def sample_layout():
    # Sample data for testing
    xs = [1, 2, 3]
    ys = [4, 5, 6]
    zs = [7, 8, 9]
    return Layout(xs, ys, zs)


def test_layout_creation(sample_layout):
    assert np.array_equal(sample_layout.x, np.array([1, 2, 3]))
    assert np.array_equal(sample_layout.y, np.array([4, 5, 6]))
    assert np.array_equal(sample_layout.z, np.array([7, 8, 9]))


def test_default_z():
    layout = Layout([1, 2, 3], [4, 5, 6])
    assert np.array_equal(layout.z, np.zeros_like(np.array([1, 2, 3])))


def test_rotate_origin(sample_layout):
    rotated_layout = sample_layout.rotate(90, center="origin")
    assert np.allclose(rotated_layout.x, -sample_layout.y)
    assert np.allclose(rotated_layout.y, sample_layout.x)


def test_rotate_centroid(sample_layout):
    rotated_layout = sample_layout.rotate(90, center="centroid")
    assert np.allclose(rotated_layout.x, [3, 2, 1])
    assert np.allclose(rotated_layout.y, rotated_layout.y)


def test_iter_downstream():
    layout = Layout([3, 1, 2], [6, 4, 5], [9, 7, 8])
    sorted_layout = list(layout.iter_downstream())
    expected_sorted_layout = [(1, (1, 4, 7)), (2, (2, 5, 8)), (0, (3, 6, 9))]
    assert sorted_layout == expected_sorted_layout
