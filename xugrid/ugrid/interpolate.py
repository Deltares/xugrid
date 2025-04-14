from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, NamedTuple, Tuple

import numba as nb
import numpy as np
import xarray as xr
from scipy import sparse

from xugrid.constants import FloatArray, IntArray
from xugrid.core.sparse import MatrixCSR, columns_and_values, nzrange, row_slice


@nb.njit(inline="always")
def lower_slice(ilu, row: int) -> slice:
    return slice(ilu.indptr[row], ilu.uptr[row])


@nb.njit(inline="always")
def upper_slice(ilu, row: int) -> slice:
    return slice(ilu.uptr[row], ilu.indptr[row + 1])


@nb.njit
def set_uptr(ilu: ILU0Preconditioner) -> None:
    # i is row index, j is column index
    for i in range(ilu.n):
        for nzi in nzrange(ilu, i):
            j = ilu.indices[nzi]
            if j > i:
                ilu.uptr[i] = nzi
                break
    return


@nb.njit
def _update(ilu: ILU0Preconditioner, A: MatrixCSR, delta: float, relax: float):
    """
    Perform zero fill-in incomplete lower-upper (ILU0) factorization
    using the values of A.
    """
    ilu.work[:] = 0.0
    visited = np.full(ilu.n, False)

    # i is row index, j is column index, v is value.
    for i in range(ilu.n):
        for j, v in columns_and_values(A, row_slice(A, i)):
            visited[j] = True
            ilu.work[j] += v

        rs = 0.0
        for j in ilu.indices[lower_slice(ilu, i)]:
            # Compute row multiplier
            multiplier = ilu.work[j] * ilu.diagonal[j]
            ilu.work[j] = multiplier
            # Perform linear combination
            for jj, vv in columns_and_values(ilu, upper_slice(ilu, j)):
                if visited[jj]:
                    ilu.work[jj] -= multiplier * vv
                else:
                    rs += multiplier * vv

        diag = ilu.work[i]
        multiplier = (1.0 + delta) * diag - (relax * rs)
        # Work around a zero-valued pivot and make sure the multiplier hasn't
        # changed sign.
        if multiplier == 0:
            multiplier = 1e-6
        elif np.sign(multiplier) != np.sign(diag):
            multiplier = np.sign(diag) * 1.0e-6
        ilu.diagonal[i] = 1.0 / multiplier

        # Reset work arrays, assign off-diagonal values
        visited[i] = False
        ilu.work[i] = 0.0
        for nzi in nzrange(ilu, i):
            j = ilu.indices[nzi]
            ilu.data[nzi] = ilu.work[j]
            ilu.work[j] = 0.0
            visited[j] = False

    return


@nb.njit
def _solve(ilu: ILU0Preconditioner, r: np.ndarray):
    r"""
    LU \ r

    Stores the result in the pre-allocated work array.
    """
    ilu.work[:] = 0.0

    # forward
    for i in range(ilu.n):
        value = r[i]
        for j, v in columns_and_values(ilu, lower_slice(ilu, i)):
            value -= v * ilu.work[j]
        ilu.work[i] = value

    # backward
    for i in range(ilu.n - 1, -1, -1):
        value = ilu.work[i]
        for j, v in columns_and_values(ilu, upper_slice(ilu, i)):
            value -= v * ilu.work[j]
        ilu.work[i] = value * ilu.diagonal[i]

    return


class ILU0Preconditioner(NamedTuple):
    """
    Preconditioner based on zero fill-in lower-upper (ILU0) factorization.

    Data is stored in compressed sparse row (CSR) format. The diagonal
    values have been extracted for easier access. Upper and lower values
    are stored in CSR format. Next to the indptr array, which identifies
    the start and end of each row, the uptr array has been added to
    identify the start to the right of the diagonal. In case the row to the
    right of the diagonal is empty, it contains the end of the rows as
    indicated by the indptr array.

    Parameters
    ----------
    n: int
        Number of rows
    m: int
        Number of columns
    indptr: np.ndarray of int
        CSR format index pointer array of the matrix
    uptr: np.ndarray of int
        CSR format index pointer array of the upper elements (diagonal or higher)
    indices: np.ndarray of int
        CSR format index array of the matrix
    data: np.ndarray of float
        CSR format data array of the matrix
    diagonal: np.ndarray of float
        Diagonal values of LU factorization
    work: np.ndarray of float
        Work array. Used in factorization and solve.
    """

    n: int
    m: int
    indptr: IntArray
    uptr: IntArray
    indices: IntArray
    data: FloatArray
    diagonal: FloatArray
    work: FloatArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.m)

    @property
    def dtype(self):
        return self.data.dtype

    @staticmethod
    def from_csr_matrix(
        A: sparse.csr_matrix, delta: float = 0.0, relax: float = 0.0
    ) -> "ILU0Preconditioner":
        # Create a copy of the sparse matrix with the diagonals removed.
        n, m = A.shape
        coo = A.tocoo()
        i = coo.row
        j = coo.col
        offdiag = i != j
        ii = i[offdiag]
        indices = j[offdiag]
        indptr = sparse.csr_matrix((indices, (ii, indices)), shape=A.shape).indptr

        ilu = ILU0Preconditioner(
            n=n,
            m=m,
            indptr=indptr,
            uptr=indptr[1:].copy(),
            indices=indices,
            data=np.empty(indices.size),
            diagonal=np.empty(n),
            work=np.empty(n),
        )
        set_uptr(ilu)

        _update(ilu, MatrixCSR.from_csr_matrix(A), delta, relax)
        return ilu

    def update(self, A, delta=0.0, relax=0.0) -> None:
        _update(self, MatrixCSR.from_csr_matrix(A), delta, relax)
        return

    def matvec(self, r) -> FloatArray:
        _solve(self, r)
        return self.work

    def __repr__(self) -> str:
        return f"ILU0Preconditioner of type {self.dtype} and shape {self.shape}"


def laplace_interpolate(
    data: FloatArray,
    connectivity: sparse.csr_matrix,
    use_weights: bool,
    direct_solve: bool = False,
    delta=0.0,
    relax=0.0,
    atol: float = 0.0,
    rtol: float = 1.0e-5,
    maxiter: int = 500,
):
    """
    Fill gaps in ``data`` (``np.nan`` values) using Laplace interpolation.

    This solves Laplace's equation where there is no data, with data values
    functioning as fixed potential boundary conditions.

    Note that an iterative solver method will be required for large grids.
    Refer to the documentation of :py:func:`scipy.sparse.linalg.cg`.

    Parameters
    ----------
    data: ndarray of floats with shape ``(n,)``
    connectivity: scipy.sparse.csr_matrix with shape ``(n, n)``
        Sparse connectivity matrix containing ``n_nonzero`` indices and weight values.
    use_weights: bool, default False.
        Wether to use the data attribute of the connectivity matrix as
        coefficients. If ``False``, defaults to uniform coefficients of 1.
    direct_solve: bool, optional, default ``False``
        Whether to use a direct or an iterative solver or a conjugate gradient
        solver. Direct methods provides an exact answer, but are unsuitable
        for large problems.
    delta: float, default 0.0
        ILU0 preconditioner non-diagonally dominant correction.
    relax: float, default 0.0
        Modified ILU0 preconditioner relaxation factor.
    atol: float, optional, default 0.0
        Convergence tolerance for ``scipy.sparse.linalg.cg``.
    tol: float, optional, default 1.0e-5.
        Convergence tolerance for ``scipy.sparse.linalg.cg``.
    maxiter: int, default 500.
        Maximum number of iterations for ``scipy.sparse.linalg.cg``.

    Returns
    -------
    filled: ndarray of floats
    """
    # Input checks
    n, m = connectivity.shape
    if n != m:
        raise ValueError(f"connectivity is not a square matrix: ({n}, {m})")
    if data.shape != (n,):
        raise ValueError(f"expected data of shape ({n},), received: {data.shape}")

    # Find the elements with data
    variable = np.isnan(data)
    constant = ~variable
    if variable.all():
        raise ValueError("data is fully nodata")
    elif constant.all():
        return data.copy()

    coo = connectivity.tocoo()
    i = coo.row
    j = coo.col

    # Create uniform weighting if sparse data is not to be used.
    if use_weights:
        weights = connectivity.data
    else:
        weights = np.ones(i.size)

    # Find the connections to constant cells and the (non-)zero values in the
    # coefficient matrix.
    constant_j = constant[j]
    nonzero = ~(constant[i] | constant[j])

    # Build the right-hand-side and the diagonal.
    rhs_coo_content = (
        weights[constant_j],
        (i[constant_j], j[constant_j]),
    )
    rhs = -sparse.csr_matrix(rhs_coo_content, shape=(n, n)).dot(data)
    diag_coo_content = (weights, (i, j))
    diagonal = -sparse.csr_matrix(diag_coo_content, shape=(n, n)).sum(axis=1).A1
    # Create the diagonal numbering.
    ii = np.arange(n)

    # Set the constant values via the diagonal and the rhs.
    diagonal[constant] = 1.0
    rhs[constant] = data[constant]
    # Remove all zero values and add the diagonal.
    weights = np.concatenate([weights[nonzero], diagonal])
    i = np.concatenate([i[nonzero], ii])
    j = np.concatenate([j[nonzero], ii])
    coo_content = (weights, (i, j))
    A = sparse.csr_matrix(coo_content, shape=(n, n))

    if direct_solve:
        x = sparse.linalg.spsolve(A, rhs)
    else:
        # Create preconditioner M
        M = ILU0Preconditioner.from_csr_matrix(A, delta=delta, relax=relax)
        # Call conjugate gradient solver
        x, info = sparse.linalg.cg(A, rhs, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        if info < 0:
            raise ValueError("scipy.sparse.linalg.cg: illegal input or breakdown")
        elif info > 0:
            warnings.warn(f"Failed to converge after {maxiter} iterations")

    return x


def interpolate_na_helper(
    da: xr.DataArray,
    ugrid_dim: str,
    func: Callable,
    kwargs: Dict[str, Any],
):
    """Use apply ufunc to broadcast over the non UGRID dims."""
    da_filled = xr.apply_ufunc(
        func,
        da,
        input_core_dims=[[ugrid_dim]],
        output_core_dims=[[ugrid_dim]],
        vectorize=True,
        kwargs=kwargs,
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[da.dtype],
    )
    return da_filled
