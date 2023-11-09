"""Contains common reduction methods."""
import numpy as np


def mean(values, indices, weights):
    vsum = 0.0
    wsum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        vsum += w * v
        wsum += w
    if wsum == 0:
        return np.nan
    else:
        return vsum / wsum


def harmonic_mean(values, indices, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v) or v == 0:
            continue
        if w > 0:
            w_sum += w
            v_agg += w / v
    if v_agg == 0 or w_sum == 0:
        return np.nan
    else:
        return w_sum / v_agg


def geometric_mean(values, indices, weights):
    v_agg = 0.0
    w_sum = 0.0

    # Compute sum to normalize weights to avoid tiny or huge values in exp
    normsum = 0.0
    for i, w in zip(indices, weights):
        normsum += w
    # Early return if no values
    if normsum == 0:
        return np.nan

    for i, w in zip(indices, weights):
        w = w / normsum
        v = values[i]
        # Skip if v is NaN or 0.
        if v > 0 and w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
        elif v < 0:
            # Computing a geometric mean of negative numbers requires a complex
            # value.
            return np.nan

    if w_sum == 0:
        return np.nan
    else:
        # w_sum is generally 1.0, but might not be if there are NaNs present!
        return np.exp((1.0 / w_sum) * v_agg)


def sum(values, indices, weights):
    v_sum = 0.0
    w_sum = 0.0

    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        v_sum += v
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_sum


def minimum(values, indices, weights):
    vmin = values[indices[0]]
    for i in indices:
        v = values[i]
        if np.isnan(v):
            continue
        if v < vmin:
            vmin = v
    return vmin


def maximum(values, indices, weights):
    vmax = values[indices[0]]
    for i in indices:
        v = values[i]
        if np.isnan(v):
            continue
        if v > vmax:
            vmax = v
    return vmax


def mode(values, indices, weights):
    # Area weighted mode. We use a linear search to accumulate weights, as we
    # generally expect a relatively small number of elements in the indices and
    # weights arrays.
    accum = weights.copy()
    w_sum = 0
    for running_total, (i, w) in enumerate(zip(indices, weights)):
        v = values[i]
        if np.isnan(v):
            continue
        w_sum += 1
        for j in range(running_total):  # Compare with previously found values
            if values[j] == v:  # matches previous value
                accum[j] += w  # increase previous weight sum
                break

    if w_sum == 0:  # It skipped everything: only nodata values
        return np.nan
    else:  # Find value with highest frequency
        w_max = 0
        for i in range(accum.size):
            w_accum = accum[i]
            if w_accum > w_max:
                w_max = w_accum
                v = values[i]
        return v


def median(values, indices, weights):
    # TODO: more efficient implementation?
    # See: https://github.com/numba/numba/blob/0441bb17c7820efc2eba4fd141b68dac2afa4740/numba/np/arraymath.py#L1693
    return np.nanpercentile(values[indices], 50)


def first_order_conservative(values, indices, weights):
    # Uses relative weights!
    # Rename to: first order conservative?
    v_agg = 0.0
    w_sum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        v_agg += v * w
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_agg


conductance = first_order_conservative


def max_overlap(values, indices, weights):
    max_w = 0.0
    v = np.nan
    for i, w in zip(indices, weights):
        if w > max_w:
            max_w = w
            v_temp = values[i]
            if np.isnan(v_temp):
                continue
            v = v_temp
    return v


ASBOLUTE_OVERLAP_METHODS = {
    "mean": mean,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
    "sum": sum,
    "minimum": minimum,
    "maximum": maximum,
    "mode": mode,
    "median": median,
    "max_overlap": max_overlap,
}


RELATIVE_OVERLAP_METHODS = {
    "conductance": conductance,
    "first_order_conservative": first_order_conservative,
}
