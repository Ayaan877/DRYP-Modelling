# Locally weighted regression (LOESS) time series interpolation.
#
# Callable:
#   lwr_interpolate(x, y, x_new=None, step=1, poly_order=2, n_neighbours=7,
#                   max_gap=None, clip_min=None, clip_max=None)
#
# Inputs:
#   x            observation times (numbers, datetime64 or date objects), shape (n,)
#   y            observations, shape (n, ...); time on axis 0, NaN = missing
#   x_new        optional output times (default: regular grid of `step` from first to last x)
#   step         grid spacing when x_new is None (days for dates)
#   poly_order   degree of the local polynomial
#   n_neighbours nearest valid observations used per fit
#   max_gap      gaps between bracketing observations wider than this stay NaN
#   clip_min/max bounds for interpolated values (observed values untouched)
#
# Outputs:
#   x_new, y_new   output times (days for dates) and the filled series, shape (len(x_new), ...)
#                  NaN outside the first/last valid observation or in skipped gaps

import numpy as np


def to_numbers(x):
	x = np.asarray(x)
	if np.issubdtype(x.dtype, np.datetime64):
		return x.astype("datetime64[D]").astype(np.float64)
	if x.dtype == object:
		return np.array([v.toordinal() if hasattr(v, "toordinal") else v for v in x], dtype=np.float64)
	return x.astype(np.float64)


def fill_series(x, y, x_new, poly_order, n_neighbours, max_gap, clip_min, clip_max):
	out = np.full(x_new.size, np.nan)
	valid = ~np.isnan(y)
	vx, vy = x[valid], y[valid]
	k = max(n_neighbours, poly_order + 1)
	if vx.size < max(poly_order + 1, 2):
		return out
	pos = np.searchsorted(vx, x_new)
	for i, target in enumerate(x_new):
		p = pos[i]
		if target < vx[0] or target > vx[-1]:
			continue
		if vx[p] == target:
			out[i] = vy[p]
			continue
		if max_gap is not None and vx[p] - vx[p - 1] > max_gap:
			continue
		lo, hi = max(0, p - k), min(vx.size, p + k)
		near = np.argsort(np.abs(vx[lo:hi] - target), kind="stable")[:k] + lo
		dx = vx[near] - target
		d = np.abs(dx)
		weights = (1 - (d / (d.max() * 1.05)) ** 3) ** 3
		root_w = np.sqrt(weights)
		design = np.vander(dx, poly_order + 1, increasing=True)
		value = np.linalg.lstsq(design * root_w[:, None], vy[near] * root_w, rcond=None)[0][0]
		if clip_min is not None:
			value = max(value, clip_min)
		if clip_max is not None:
			value = min(value, clip_max)
		out[i] = value
	return out


def lwr_interpolate(x, y, x_new=None, step=1, poly_order=2, n_neighbours=7,
					max_gap=None, clip_min=None, clip_max=None):
	x = to_numbers(x)
	y = np.asarray(y, dtype=np.float64)
	if y.shape[0] != x.size:
		raise ValueError("y must have the same length as x along axis 0")
	order = np.argsort(x, kind="stable")
	x, y = x[order], y[order]
	if np.any(np.diff(x) == 0):
		raise ValueError("x contains duplicate times")
	if x_new is None:
		x_new = np.arange(x[0], x[-1] + step / 2, step)
	else:
		x_new = np.sort(to_numbers(x_new))
	flat = y.reshape(x.size, -1)
	out = np.full((x_new.size, flat.shape[1]), np.nan)
	for j in range(flat.shape[1]):
		if not np.isnan(flat[:, j]).all():
			out[:, j] = fill_series(x, flat[:, j], x_new, poly_order, n_neighbours, max_gap, clip_min, clip_max)
	return x_new, out.reshape((x_new.size,) + y.shape[1:])
