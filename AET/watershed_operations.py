import re

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

import watershed_data


PROJECT_CRS = CRS.from_epsg(32643)
OPERATORS = {
	"+": np.add,
	"-": np.subtract,
	"*": np.multiply,
	"/": np.divide,
}
OPERATOR_PATTERN = re.compile(r"([+\-*/])")


def resolve_request(expression, year):
	"""Load a dataset name or a left-to-right chain such as "SSEBop 30m/ref ET ERA5"."""
	parts = [part.strip(" ()") for part in OPERATOR_PATTERN.split(expression)]
	result = watershed_data.load_data(parts[0], year)
	for operator, name in zip(parts[1::2], parts[2::2]):
		result = apply_operation(result, operator, watershed_data.load_data(name, year))
	return result


def target_grid(series):
	finest = min(series, key=lambda item: item.resolution_m)
	if finest.crs == PROJECT_CRS:
		return finest.transform, finest.values.shape[2], finest.values.shape[1]
	bounds = array_bounds(finest.values.shape[1], finest.values.shape[2], finest.transform)
	transform, width, height = calculate_default_transform(
		finest.crs,
		PROJECT_CRS,
		finest.values.shape[2],
		finest.values.shape[1],
		*bounds,
	)
	return transform, width, height


def align_series(series, dates, transform, width, height):
	indexes = {value: index for index, value in enumerate(series.dates)}
	output = np.full((len(dates), height, width), np.nan, dtype=np.float32)
	for output_index, value in enumerate(dates):
		frame = series.values[indexes[value]]
		if series.crs == PROJECT_CRS and series.transform == transform and frame.shape == (height, width):
			output[output_index] = frame
			continue
		reproject(
			frame,
			output[output_index],
			src_transform=series.transform,
			src_crs=series.crs,
			src_nodata=np.nan,
			dst_transform=transform,
			dst_crs=PROJECT_CRS,
			dst_nodata=np.nan,
			resampling=Resampling.nearest,
		)
	return output


def apply_operation(left_series, operator, right_series):
	series = [left_series, right_series]
	common_dates = sorted(set(left_series.dates).intersection(right_series.dates))
	if not common_dates:
		raise ValueError("No shared observation dates between the operands")
	transform, width, height = target_grid(series)
	left = align_series(left_series, common_dates, transform, width, height)
	right = align_series(right_series, common_dates, transform, width, height)
	with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
		values = np.asarray(OPERATORS[operator](left, right), dtype=np.float32)
	values[~np.isfinite(values)] = np.nan
	same_units = left_series.units == right_series.units
	if operator in ("+", "-") and same_units:
		units = left_series.units
	elif operator == "/" and same_units:
		units = "1"
	else:
		units = f"{left_series.units} {operator} {right_series.units}"
	return watershed_data.DataSeries(
		common_dates,
		values,
		transform,
		PROJECT_CRS,
		min(item.resolution_m for item in series),
		units,
		max(item.cadence_days for item in series),
	)