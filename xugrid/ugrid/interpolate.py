import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from xugrid.constants import FloatArray


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
