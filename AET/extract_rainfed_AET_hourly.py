from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.transform import from_origin
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_NETCDF = SCRIPT_DIR / "OBF_hourly_grid.nc"
AET_VARIABLE = "aet"
TIME_VARIABLE = "time"
X_VARIABLE = "lon"
Y_VARIABLE = "lat"
DEFAULT_CRS = "EPSG:32643"

OUTPUT_DIR = SCRIPT_DIR / "rainfed_AET_hourly"
OUTPUT_GEOTIFF_DIR = OUTPUT_DIR / "GeoTIFF"
OUTPUT_NETCDF_DIR = OUTPUT_DIR / "NetCDF"
OUTPUT_PREFIX = "rainfed_AET_hourly"

NODATA = -99999.0
NETCDF_COMPRESSION_LEVEL = 4
TIME_ORIGIN = datetime(1980, 1, 1)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def coordinate_spacing(values: np.ndarray, name: str) -> float:
	if values.size < 2:
		raise ValueError(f"{name} coordinate requires at least two values")
	steps = np.diff(values)
	spacing = float(np.median(np.abs(steps)))
	if spacing == 0 or not np.allclose(np.abs(steps), spacing):
		raise ValueError(f"{name} coordinates must be regularly spaced and monotonic")
	return spacing


def netcdf_grid(
	dataset: Dataset,
) -> tuple[Any, np.ndarray, np.ndarray, rasterio.Affine, CRS]:
	for name in (AET_VARIABLE, TIME_VARIABLE, X_VARIABLE, Y_VARIABLE):
		if name not in dataset.variables:
			raise ValueError(f"Configured variable {name!r} was not found in {INPUT_NETCDF}")

	aet_variable = dataset.variables[AET_VARIABLE]
	x_variable = dataset.variables[X_VARIABLE]
	y_variable = dataset.variables[Y_VARIABLE]
	if len(x_variable.dimensions) != 1 or len(y_variable.dimensions) != 1:
		raise ValueError("AET X/Y coordinates must be one-dimensional")
	x_dimension = x_variable.dimensions[0]
	y_dimension = y_variable.dimensions[0]
	if set(aet_variable.dimensions) != {TIME_VARIABLE, y_dimension, x_dimension}:
		raise ValueError("AET variable must use the configured time, Y, and X dimensions")

	x_values = np.asarray(x_variable[:], dtype=np.float64)
	y_values = np.asarray(y_variable[:], dtype=np.float64)
	dx = coordinate_spacing(x_values, X_VARIABLE)
	dy = coordinate_spacing(y_values, Y_VARIABLE)
	if x_values[0] > x_values[-1]:
		raise ValueError("X coordinates must increase from west to east")
	transform = from_origin(
		float(x_values[0]) - dx / 2,
		float(np.max(y_values)) + dy / 2,
		dx,
		dy,
	)
	projection = getattr(dataset, "projection", DEFAULT_CRS)
	crs = CRS.from_user_input(projection)
	return aet_variable, x_values, y_values, transform, crs


def read_times(dataset: Dataset) -> list[datetime]:
	time_variable = dataset.variables[TIME_VARIABLE]
	units = getattr(time_variable, "units", None)
	if not units:
		raise ValueError(f"Time variable {TIME_VARIABLE!r} has no units")
	calendar = getattr(time_variable, "calendar", "standard")
	decoded = num2date(
		time_variable[:], units=units, calendar=calendar,
		only_use_cftime_datetimes=False,
	)
	return [
		datetime(value.year, value.month, value.day, value.hour, value.minute, value.second)
		for value in np.asarray(decoded, dtype=object).reshape(-1)
	]


def create_year_netcdf(
	path: Path,
	year_times: list[datetime],
	transform: rasterio.Affine,
	width: int,
	height: int,
	crs: CRS,
	units: str,
) -> tuple[Dataset, Any]:
	path.parent.mkdir(parents=True, exist_ok=True)
	output = Dataset(path, "w", format="NETCDF4")
	output.Conventions = "CF-1.8"
	output.title = "Hourly rainfed actual evapotranspiration"
	output.createDimension("time", len(year_times))
	output.createDimension("y", height)
	output.createDimension("x", width)
	time_var = output.createVariable("time", "i8", ("time",))
	time_var.units = f"hours since {TIME_ORIGIN:%Y-%m-%d %H:%M:%S}"
	time_var.calendar = "gregorian"
	time_var.standard_name = "time"
	time_var[:] = [(value - TIME_ORIGIN).total_seconds() / 3600 for value in year_times]
	x_var = output.createVariable("x", "f4", ("x",))
	y_var = output.createVariable("y", "f4", ("y",))
	x_var[:] = transform.c + (np.arange(width) + 0.5) * transform.a
	y_var[:] = transform.f + (np.arange(height) + 0.5) * transform.e
	x_var.axis = "X"
	y_var.axis = "Y"
	x_var.standard_name = "projection_x_coordinate"
	y_var.standard_name = "projection_y_coordinate"
	x_var.units = "m"
	y_var.units = "m"
	crs_var = output.createVariable("crs", "i4")
	crs_var.spatial_ref = crs.to_wkt()
	crs_var.crs_wkt = crs.to_wkt()
	aet_var = output.createVariable(
		AET_VARIABLE, "f4", ("time", "y", "x"), zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL, chunksizes=(1, height, width),
		fill_value=NODATA,
	)
	aet_var.units = units
	aet_var.long_name = "Hourly rainfed actual evapotranspiration"
	aet_var.grid_mapping = "crs"
	return output, aet_var


def write_geotiff(
	path: Path,
	frame: np.ndarray,
	transform: rasterio.Affine,
	crs: CRS,
	units: str,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	profile = {
		"driver": "GTiff", "dtype": "float32", "count": 1,
		"height": frame.shape[0], "width": frame.shape[1],
		"crs": crs, "transform": transform, "nodata": NODATA,
		"compress": "deflate",
	}
	with rasterio.open(path, "w", **profile) as destination:
		destination.write(np.where(np.isfinite(frame), frame, NODATA).astype(np.float32), 1)
		destination.update_tags(
			units=units,
			long_name="Hourly rainfed actual evapotranspiration",
		)


def extract_rainfed_aet_hourly() -> int:
	"""Write hourly AET frames as GeoTIFFs and one NetCDF per year."""
	if not INPUT_NETCDF.is_file():
		raise FileNotFoundError(f"Hourly OBF NetCDF not found: {INPUT_NETCDF}")
	total_written = 0
	with Dataset(INPUT_NETCDF, "r") as source:
		aet_variable, x_values, y_values, transform, crs = netcdf_grid(source)
		timestamps = read_times(source)
		if len(timestamps) != aet_variable.shape[aet_variable.dimensions.index(TIME_VARIABLE)]:
			raise ValueError("AET time/data length mismatch")
		if aet_variable.dimensions.index(TIME_VARIABLE) != 0:
			raise ValueError("AET time dimension must be first")
		units = str(getattr(aet_variable, "units", "mm/hr"))
		years: dict[int, list[int]] = {}
		for index, timestamp in enumerate(timestamps):
			years.setdefault(timestamp.year, []).append(index)
		reverse_y = y_values[0] < y_values[-1]
		height, width = len(y_values), len(x_values)

		for year, indices in tqdm(sorted(years.items()), desc="AET yearly files", unit="year"):
			year_times = [timestamps[index] for index in indices]
			netcdf_path = OUTPUT_NETCDF_DIR / f"{OUTPUT_PREFIX}_{year}.nc"
			output, output_aet = create_year_netcdf(
				netcdf_path, year_times, transform, width, height, crs, units,
			)
			try:
				for output_index, source_index in enumerate(tqdm(
					indices, desc=f"AET {year}", unit="hour", leave=False
				)):
					frame = np.ma.filled(aet_variable[source_index, :, :], np.nan).astype(np.float32)
					if reverse_y:
						frame = frame[::-1, :]
					frame[~np.isfinite(frame)] = np.nan
					stamp = timestamps[source_index].strftime("%Y%m%d%H")
					write_geotiff(
						OUTPUT_GEOTIFF_DIR / str(year) / f"{OUTPUT_PREFIX}_{stamp}.tif",
						frame, transform, crs, units,
					)
					output_aet[output_index] = np.where(np.isfinite(frame), frame, NODATA)
					total_written += 1
			finally:
				output.close()
	return total_written


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written = extract_rainfed_aet_hourly()
	print(f"Wrote {written} hourly rainfed AET GeoTIFFs and yearly NetCDFs to {OUTPUT_DIR}")