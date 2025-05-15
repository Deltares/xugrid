# %%

import numpy as np
from scipy import sparse

A = np.loadtxt("../A.txt")
rhs = np.loadtxt("../rhs.txt")
matrix = sparse.csr_matrix(A)

x, info = sparse.linalg.cg(matrix, rhs, atol=1e-9, rtol=1e-9)
xexact = sparse.linalg.spsolve(matrix, rhs)
d = x - xexact
print(d.min(), d.max())
# %%
