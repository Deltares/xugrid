import numpy as np

from xugrid.regrid.intersection import intersect_coordinates_1d


def test_intersection_1d():
    def test_commutative(a, b):
        i_ab, j_ab, o_ab = intersect_coordinates_1d(a, b)
        i_ba, j_ba, o_ba = intersect_coordinates_1d(b, a)
        assert np.array_equal(i_ab, j_ba)
        assert np.array_equal(j_ab, i_ba)
        assert np.allclose(o_ab, o_ba)

    expected_i = [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]
    expected_j = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    expected_overlap = [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0]

    # Complete range
    src_x = np.arange(0.0, 11.0, 1.0)
    dst_x = np.arange(0.0, 11.0, 2.5)
    # Returns tuples with (src_ind, dst_ind)
    i, j, overlap = intersect_coordinates_1d(src_x, dst_x)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)
    assert np.allclose(overlap, expected_overlap)
    test_commutative(src_x, dst_x)

    # Negative coords
    src_x = np.arange(-20.0, -9.0, 1.0)
    dst_x = np.arange(-20.0, -9.0, 2.5)
    i, j, overlap = intersect_coordinates_1d(src_x, dst_x)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)
    assert np.allclose(overlap, expected_overlap)
    test_commutative(src_x, dst_x)

    # Negative to postive
    src_x = np.arange(-5.0, 6.0, 1.0)
    dst_x = np.arange(-5.0, 6.0, 2.5)
    i, j, overlap = intersect_coordinates_1d(src_x, dst_x)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)
    assert np.allclose(overlap, expected_overlap)
    test_commutative(src_x, dst_x)

    # Partial dst
    src_x = np.arange(0.0, 11.0, 1.0)
    dst_x = np.arange(5.0, 11.0, 2.5)
    i, j, overlap = intersect_coordinates_1d(src_x, dst_x)
    assert np.array_equal(i, [5, 6, 7, 7, 8, 9])
    assert np.array_equal(j, [0, 0, 0, 1, 1, 1])
    assert np.allclose(overlap, [1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    test_commutative(src_x, dst_x)

    # Partial src
    src_x = np.arange(5.0, 11.0, 1.0)
    dst_x = np.arange(0.0, 11.0, 2.5)
    i, j, overlap = intersect_coordinates_1d(src_x, dst_x)
    assert np.array_equal(i, [0, 1, 2, 2, 3, 4])
    assert np.array_equal(j, [2, 2, 2, 3, 3, 3])
    assert np.allclose(overlap, [1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    test_commutative(src_x, dst_x)
