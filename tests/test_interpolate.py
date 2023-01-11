import numpy as np
import pytest
from scipy import sparse

from xugrid.ugrid import interpolate


def test_laplace_interpolate():
    i = np.array([0, 1, 1, 2, 2, 3, 3])
    j = np.array([1, 0, 2, 1, 3, 2, 4])
    coo_content = (j, (i, j))
    data = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    with pytest.raises(ValueError, match="connectivity is not a square matrix"):
        con = sparse.coo_matrix(coo_content, shape=(4, 5)).tocsr()
        interpolate.laplace_interpolate(con, data)

    expected = np.arange(1.0, 6.0)
    con = sparse.coo_matrix(coo_content, shape=(5, 5)).tocsr()
    actual = interpolate.laplace_interpolate(con, data, direct_solve=True)
    assert np.allclose(actual, expected)

    actual = interpolate.laplace_interpolate(con, data, direct_solve=False)
    assert np.allclose(actual, expected)
