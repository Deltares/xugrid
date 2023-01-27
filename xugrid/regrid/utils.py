import numpy as np


def create_linear_index(arrays, dims):
    meshgrids = [a.ravel() for a in np.meshgrid(*arrays, indexing="ij")]
    return np.ravel_multi_index(meshgrids, dims)


def create_weights(arrays):
    meshgrids = np.meshgrid(*arrays, indexing="ij")
    weight = meshgrids[0]
    for dim_weight in meshgrids[1:]:
        weight *= dim_weight
    return weight.ravel()


def broadcast(
    source_shape,
    target_shape,
    source_indices,
    target_indices,
    weights,
):
    source_index = create_linear_index(source_indices, source_shape)
    target_index = create_linear_index(target_indices, target_shape)
    weights = create_weights(weights)
    return source_index, target_index, weights


def alt_cumsum(a):
    """
    Alternative cumsum, always starts at 0 and omits the last value of the
    regular cumsum.
    """
    out = np.empty(a.size, a.dtype)
    out[0] = 0
    np.cumsum(a[:-1], out=out[1:])
    return out
