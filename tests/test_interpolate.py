import numpy as np
import pytest
from scipy import sparse

from xugrid.ugrid import interpolate


def test_ilu0():
    # Create a 1D Laplace problem:
    #
    # * Dirichlet boundary left and right of 1.0.
    # * Constant inflow 0.001 everywhere else.
    #
    n = 1000
    d = np.ones(n)
    A = sparse.diags((-d[:-1], 2 * d, -d[:-1]), (-1, 0, 1)).tocsr()
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[-1, -1] = 1.0
    A[-1, -2] = 0.0
    b = np.full(n, 0.001)
    b[0] = 1.0
    b[-1] = 1.0
    M = interpolate.ILU0Preconditioner.from_csr_matrix(A)
    _, info_cg = sparse.linalg.cg(A, b, maxiter=10)
    x_pcg, info_pcg = sparse.linalg.cg(A, b, maxiter=10, M=M)
    x_direct = sparse.linalg.spsolve(A, b)
    assert info_cg != 0  # cg does not converge
    assert info_pcg == 0  # preconditioned cg does converge
    assert np.allclose(x_pcg, x_direct)  # answer matches direct solve


def test_laplace_interpolate():
    i = np.array([0, 1, 1, 2, 2, 3, 3])
    j = np.array([1, 0, 2, 1, 3, 2, 4])
    coo_content = (j, (i, j))
    data = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    with pytest.raises(ValueError, match="connectivity is not a square matrix"):
        con = sparse.coo_matrix(coo_content, shape=(4, 5)).tocsr()
        interpolate.laplace_interpolate(data, con, use_weights=False)

    expected = np.arange(1.0, 6.0)
    con = sparse.coo_matrix(coo_content, shape=(5, 5)).tocsr()
    actual = interpolate.laplace_interpolate(
        data, con, use_weights=False, direct_solve=True
    )
    assert np.allclose(actual, expected)

    actual = interpolate.laplace_interpolate(
        data, con, use_weights=False, direct_solve=False
    )
    assert np.allclose(actual, expected)


def test_nearest_interpolate():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.zeros_like(x)
    coordinates = np.column_stack((x, y))
    data = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
    actual = interpolate.nearest_interpolate(data, coordinates, np.inf)
    assert np.allclose(actual, np.array([0.0, 0.0, 0.0, 4.0, 4.0]))

    actual = interpolate.nearest_interpolate(data, coordinates, 1.1)
    assert np.allclose(actual, np.array([0.0, 0.0, np.nan, 4.0, 4.0]), equal_nan=True)

    with pytest.raises(ValueError, match="All values are NA."):
        interpolate.nearest_interpolate(np.full_like(data, np.nan), coordinates, np.inf)
