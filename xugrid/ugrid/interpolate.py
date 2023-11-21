import warnings
from typing import NamedTuple, Tuple

import numpy as np
import numba
import scipy.sparse
import scipy.sparse.linalg

from xugrid.constants import FloatArray, IntArray


DEM6 = 1.0e-6


class MatrixCSR(NamedTuple): 
    data: FloatArray
    indices: IntArray
    indptr: IntArray
    n: int
    nnz: int
    

class ILU0(NamedTuple):
    lower: MatrixCSR
    upper: MatrixCSR
    lower_map: IntArray
    upper_map: IntArray
    work: FloatArray
    

def create_IL0(n, indptr, indices, data):
    pass
    

class SymmetricILU0:
    def __init__(self, A: scipy.sparse.csr_matrix):
        self.L = self._create_ILU0(
            A.n,
            A.indptr,
            A.indices,
            A.data,
        )
        
        
    def solve(self, v): 
        """
        Parameters
        ----------
        v: np.ndarray
            Dense vector. Called as matvec by scipy LinearOperator.
            
        Returns
        -------
        d: np.ndarray
            Solution of LU \ v
        """
        d = np.empty_like(v)
        self._solve(self.L, v, d)
        return d
    
    @staticmethod
    @numba.njit
    def _create_ILU0(n, indptr, indices, data):
        delta = 0.0
 
    @staticmethod
    @numba.njit
    def _solve(LU: MatrixCSR, v: FloatArray, d: FloatArray) -> None:
        # Forward solve: L * d = v
        for row_index in range(LU.n):
            value = v[row_index]
            column_start = LU.indptr[row_index]
            index_upper = LU.indices[row_index]
            for linear_index in range(column_start, index_upper):
                column_index = LU.indices[linear_index]
                value -= LU.data[linear_index] * d[column_index]
            d[row_index] = value
            
        # Backward solve: d = d / U
        for row_index in range(LU.n, -1, -1):
            value = v[row_index]
            column_end = LU.indptr[row_index]
            index_upper = LU.indices[row_index]
            for linear_index in range(index_upper, column_end):
                column_index = LU.indices[linear_index]
                value -= LU.data[linear_index] * d[column_index]
            
            # Compute diagonal: d = d / U
            d[row_index] = value * LU.data[row_index]
            
        return


@numba.njit(inline="always")
def linear_indices(A: MatrixCSR, row: int) -> range:
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return range(start, end)
    

@numba.njit(inline="always")
def colvals(A: MatrixCSR, row: int) -> IntArray:
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return A.indices[start: end]


@numba.njit(inline="always")
def nzrange(A: MatrixCSR, row: int) -> Tuple[IntArray, FloatArray]:
    """Return the indices and values of a single row."""
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return A.indices[start:end], A.data[start:end]


def create_ILU0(A: MatrixCSR):
    n_lower = 0
    n_upper = 0
    # Determine the number of elements in lower and upper
    for i in range(A.n):
        for j in colvals(A, i):
            if j > i:
                n_lower += 1
            else:
                n_upper += 1

    lower = MatrixCSR(
        data=np.empty(n_lower, dtype=float),
        indices=np.zeros(n_lower, dtype=int),
        indptr=np.zeros(A.n + 1, dtype=int),
        n = A.n,
        nnz = n_lower,
    )
    lower_map = np.empty(n_lower, dtype=int)

    upper = MatrixCSR(
        data=np.empty(n_upper, dtype=float),
        indices=np.zeros(n_upper, dtype=int),
        indptr=np.zeros(A.n + 1, dtype=int),
        n = A.n,
        nnz = n_upper,
    )
    upper_map = np.empty(n_upper, dtype=int)
    
    nzi_lower = 0
    nzi_upper = 0
    for i in range(A.n):
        lower.indptr[i + 1] = lower.indptr[i]
        upper.indptr[i + 1] = upper.indptr[i]
        for nzi in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[nzi]
            if j > i:
                lower.indptr[i + 1] += 1
                lower.indices[nzi_lower] = A.data[nzi]
                lower_map[nzi_lower] = nzi
                nzi_lower += 1
            else:
                upper.indptr[i + 1] += 1
                upper.indices[nzi_upper] = A.data[nzi]
                upper_map[nzi_upper] = nzi
                nzi_upper += 1
                
    return ILU0(
        lower=lower,
        upper=upper,
        lower_map=lower_map,
        upper_map=upper_map,
    )
    

def update_ilu0(ilu0: ILU0, A: MatrixCSR):
    U = ilu0.upper
    L = ilu0.lower
    
    # Update the lower and upper matrices
    for nzi in range(L.nnz):
        L.data[i] = A.data[ilu0.lower_map[nzi]]
    for nzi in range(U.nnz):
        U.data[i] = A.data[ilu0.upper_map[nzi]]

    for i in range(A.n):
        # Get the diagonal value
        nzi = L.indptr[i + 1] - 1
        m_inverse = 1.0 / L.data[nzi]
        # Update the lower matrix
        for nzi in linear_indices(L, row=i):
            L.data[nzi] *= m_inverse
        # Update the upper matrix
        for nzi in linear_indices(U, row=i):
            multiplier = U.data[nzi]
            qn = nzi + 1
            rn = L.indptr[i + 1]
            pn = L.indptr[U.indices[nzi]]
            
        
            

    
def relaxing_ilu0(
    matrix: scipy.sparse.csr_matrix,
    preconditioned_matrix: scipy.sparse.csr_matrix,
    visited: np.ndarray,
    work: np.ndarray,
    relax: float,
    delta: float,
):
    visited[:] = False
    work[:] = 0.0
    
    nrow, _ = matrix.shape
    
    # Main loop
    for i in range(nrow):
        for j, v in nzrange(matrix, i):
            visited[j] = True
            work[j] += v
    
        # Lower loop
        rs = 0.0
        for j, v in nzrange(preconditioned_matrix, i):
            tl = work[j] * v
            work[j] = tl
            
            for jj in range(iiu, iic1):
                jjcol = preconditioned_matrix.indptr[jj]
                if visited[jj]:
                    work[jjcol] -= tl * preconditioned_matrix.data[jj]
                else:
                    rs += tl * preconditioned_matrix.data[jj]
            
        # Calculate inverse of diagonal
        d = work[i]
        tl = (1.0 + delta) * d - (relax * rs)
        signed_d = np.sign(tl) * d 
        
        if signed_d != d or abs(tl) == 0.0:
            tl = np.sign(d) * DEM6

        preconditioned_matrix.data[i] = 1.0 / tl 
        
        visited[i] = False
        work[i] = 0.0
        for j in range(ic0, ic1):
            jcol = preconditioned_matrix.indptr[j]
            preconditioned_matrix.matrix[j] = work[jcol]
            visited[jcol] = False
            work[jcol] = 0.0
            
        return


def laplace_interpolate(
    connectivity: scipy.sparse.csr_matrix,
    data: FloatArray,
    use_weights: FloatArray = None,
    direct_solve: bool = False,
    drop_tol: float = None,
    fill_factor: float = None,
    drop_rule: str = None,
    options: dict = None,
    tol: float = 1.0e-5,
    maxiter: int = 250,
):
    """
    Fill gaps in ``data`` (``np.nan`` values) using Laplace interpolation.

    This solves Laplace's equation where where there is no data, with data
    values functioning as fixed potential boundary conditions.

    Note that an iterative solver method will be required for large grids. In
    this case, some experimentation with the solver settings may be required to
    find a converging solution of sufficient accuracy. Refer to the
    documentation of :py:func:`scipy.sparse.linalg.spilu` and
    :py:func:`scipy.sparse.linalg.cg`.

    Parameters
    ----------
    connectivity: scipy.sparse.csr_matrix with shape ``(n, n)``
        Sparse connectivity matrix containing ``n_nonzero`` indices and weight values.
    data: ndarray of floats with shape ``(n,)``
    use_weights: bool, default False.
        Wether to use the data attribute of the connectivity matrix as
        coefficients. If ``False``, defaults to uniform coefficients of 1.
    direct_solve: bool, optional, default ``False``
        Whether to use a direct or an iterative solver or a conjugate gradient
        solver. Direct methods provides an exact answer, but are unsuitable
        for large problems.
    drop_tol: float, optional, default None.
        Drop tolerance for ``scipy.sparse.linalg.spilu`` which functions as a
        preconditioner for the conjugate gradient solver.
    fill_factor: float, optional, default None.
        Fill factor for ``scipy.sparse.linalg.spilu``.
    drop_rule: str, optional default None.
        Drop rule for ``scipy.sparse.linalg.spilu``.
    options: dict, optional, default None.
        Remaining other options for ``scipy.sparse.linalg.spilu``.
    tol: float, optional, default 1.0e-5.
        Convergence tolerance for ``scipy.sparse.linalg.cg``.
    maxiter: int, default 250.
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
    rhs = -scipy.sparse.csr_matrix(rhs_coo_content, shape=(n, n)).dot(data)
    diag_coo_content = (weights, (i, j))
    diagonal = -scipy.sparse.csr_matrix(diag_coo_content, shape=(n, n)).sum(axis=1).A1
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
    A = scipy.sparse.csr_matrix(coo_content, shape=(n, n))

    if direct_solve:
        x = scipy.sparse.linalg.spsolve(A, rhs)
    else:
        # Create preconditioner M
        ilu = scipy.sparse.linalg.spilu(
            A.tocsc(),
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            drop_rule=drop_rule,
            options=options,
        )
        M = scipy.sparse.linalg.LinearOperator((n, n), ilu.solve)
        # Call conjugate gradient solver
        x, info = scipy.sparse.linalg.cg(
            A, rhs, tol=tol, maxiter=maxiter, M=M, atol="legacy"
        )
        if info < 0:
            raise ValueError("scipy.sparse.linalg.cg: illegal input or breakdown")
        elif info > 0:
            warnings.warn(f"Failed to converge after {maxiter} iterations")

    return x
