from datetime import datetime, timedelta
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
VERSION = "v2"
SATELLITE_FILE = SCRIPT_DIR / f"Watershed AET hourly 250m {VERSION}" / f"watershed_AET_hourly_{VERSION}_250m_all_years.nc"
SATELLITE_AET_VARIABLE = "aet_250m"
SATELLITE_X_VARIABLE = "x_250m"
SATELLITE_Y_VARIABLE = "y_250m"
SATELLITE_CRS_VARIABLE = "crs_250m"
RAINFED_INPUT_DIR = SCRIPT_DIR / "Watershed_rainfed_AET_hourly_250m"
RAINFED_PREFIX = "rainfed_AET_hourly"

OUTPUT_DIR = SCRIPT_DIR / f"Watershed irrigation AET hourly 250m {VERSION}"
OUTPUT_PREFIX = "irrigation_AET_hourly_250m"
COMBINED_NETCDF_FILE = OUTPUT_DIR / f"{OUTPUT_PREFIX}_all_years.nc"

SOURCE_AET_VARIABLE = "aet"
IRRIGATION_VARIABLE = "irr"
TIME_VARIABLE = "time"
X_VARIABLE = "x"
Y_VARIABLE = "y"
CRS_VARIABLE = "crs"
DEFAULT_CRS = "EPSG:32643"
NODATA = -9999.0
NETCDF_COMPRESSION_LEVEL = 4
TIME_ORIGIN = datetime(2015, 1, 1)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def input_year_files() -> dict[int, Path]:
	"""Find yearly rainfed NetCDF inputs by year."""
	if not SATELLITE_FILE.is_file():
		raise FileNotFoundError(f"Satellite NetCDF was not found: {SATELLITE_FILE}")
	rainfed_files: dict[int, Path] = {}
	for path in sorted(RAINFED_INPUT_DIR.glob(f"{RAINFED_PREFIX}_*.nc")):
		try:
			year = int(path.stem.rsplit("_", 1)[1])
		except (IndexError, ValueError) as error:
			raise ValueError(f"Could not determine year from {path.name}") from error
		if year in rainfed_files:
			raise ValueError(f"Duplicate rainfed NetCDF for {year}: {path}")
		rainfed_files[year] = path
	if not rainfed_files:
		raise FileNotFoundError(f"No rainfed yearly NetCDFs found in {RAINFED_INPUT_DIR}")
	return rainfed_files


def coordinate_spacing(values: np.ndarray, name: str) -> float:
	if values.size < 2:
		raise ValueError(f"{name} coordinate requires at least two values")
	steps = np.diff(values)
	spacing = float(np.median(np.abs(steps)))
	if spacing == 0 or not np.allclose(np.abs(steps), spacing):
		raise ValueError(f"{name} coordinates must be regularly spaced and monotonic")
	return spacing


def source_metadata(
	dataset: Dataset, path: Path,
	aet_name: str, x_name: str, y_name: str, crs_name: str,
) -> tuple[Any, list[datetime], np.ndarray, np.ndarray, rasterio.Affine, CRS]:
	for name in (aet_name, TIME_VARIABLE, x_name, y_name):
		if name not in dataset.variables:
			raise ValueError(f"Configured variable {name!r} was not found in {path}")

	aet_variable = dataset.variables[aet_name]
	time_variable = dataset.variables[TIME_VARIABLE]
	x_variable = dataset.variables[x_name]
	y_variable = dataset.variables[y_name]
	if aet_variable.dimensions != (TIME_VARIABLE, y_name, x_name):
		raise ValueError(f"{aet_name!r} must use dimensions (time, y, x) in {path}")
	if x_variable.dimensions != (x_name,) or y_variable.dimensions != (y_name,):
		raise ValueError(f"X/Y coordinates must be one-dimensional in {path}")

	x_values = np.asarray(x_variable[:], dtype=np.float64)
	y_values = np.asarray(y_variable[:], dtype=np.float64)
	dx = coordinate_spacing(x_values, X_VARIABLE)
	dy = coordinate_spacing(y_values, Y_VARIABLE)
	if x_values[0] > x_values[-1]:
		raise ValueError(f"X coordinates must increase from west to east in {path}")
	transform = from_origin(
		float(x_values[0]) - dx / 2,
		float(np.max(y_values)) + dy / 2,
		dx,
		dy,
	)
	crs_variable = dataset.variables.get(crs_name)
	projection = (
		getattr(crs_variable, "spatial_ref", None)
		or getattr(crs_variable, "crs_wkt", None)
		or getattr(dataset, "projection", DEFAULT_CRS)
	)
	crs = CRS.from_user_input(projection)
	units = getattr(time_variable, "units", None)
	if not units:
		raise ValueError(f"Time variable {TIME_VARIABLE!r} has no units in {path}")
	decoded = num2date(
		time_variable[:], units=units,
		calendar=getattr(time_variable, "calendar", "standard"),
		only_use_cftime_datetimes=False,
	)
	timestamps = [
		datetime(value.year, value.month, value.day, value.hour, value.minute, value.second)
		for value in np.asarray(decoded, dtype=object).reshape(-1)
	]
	if len(timestamps) != aet_variable.shape[0]:
		raise ValueError(f"AET time/data length mismatch in {path}")
	return aet_variable, timestamps, x_values, y_values, transform, crs


def validate_grids(
	satellite_x: np.ndarray, satellite_y: np.ndarray,
	rainfed_x: np.ndarray, rainfed_y: np.ndarray,
) -> None:
	"""Confirm the satellite and rainfed inputs share one grid (no resampling is applied)."""
	if satellite_x.shape != rainfed_x.shape or satellite_y.shape != rainfed_y.shape:
		raise ValueError("Satellite and rainfed grid shapes differ")
	if not (np.allclose(satellite_x, rainfed_x, rtol=0, atol=0.5)
		and np.allclose(satellite_y, rainfed_y, rtol=0, atol=0.5)):
		raise ValueError("Satellite and rainfed grid coordinates differ")


def create_combined_netcdf(
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
	output.title = "Hourly irrigation actual evapotranspiration"
	output.createDimension(TIME_VARIABLE, len(year_times))
	output.createDimension(Y_VARIABLE, height)
	output.createDimension(X_VARIABLE, width)
	time_var = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
	time_var.units = f"hours since {TIME_ORIGIN:%Y-%m-%d %H:%M:%S}"
	time_var.calendar = "gregorian"
	time_var.standard_name = "time"
	time_var[:] = [(value - TIME_ORIGIN).total_seconds() / 3600 for value in year_times]
	x_var = output.createVariable(X_VARIABLE, "f4", (X_VARIABLE,))
	y_var = output.createVariable(Y_VARIABLE, "f4", (Y_VARIABLE,))
	x_var[:] = transform.c + (np.arange(width) + 0.5) * transform.a
	y_var[:] = transform.f + (np.arange(height) + 0.5) * transform.e
	x_var.axis = "X"
	y_var.axis = "Y"
	x_var.standard_name = "projection_x_coordinate"
	y_var.standard_name = "projection_y_coordinate"
	x_var.units = "m"
	y_var.units = "m"
	crs_var = output.createVariable(CRS_VARIABLE, "i4")
	crs_var.spatial_ref = crs.to_wkt()
	crs_var.crs_wkt = crs.to_wkt()
	irrigation_var = output.createVariable(
		IRRIGATION_VARIABLE, "f4", (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE), zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL, chunksizes=(1, height, width),
		fill_value=NODATA,
	)
	irrigation_var.units = units
	irrigation_var.long_name = "Hourly irrigation actual evapotranspiration"
	irrigation_var.grid_mapping = CRS_VARIABLE
	return output, irrigation_var


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
			long_name="Hourly irrigation actual evapotranspiration",
		)


def year_timestamps(year: int) -> list[datetime]:
	start = datetime(year, 1, 1)
	next_year = datetime(year + 1, 1, 1)
	return [
		start + timedelta(hours=hour)
		for hour in range(int((next_year - start).total_seconds() // 3600))
	]


def calculate_irrigation_aet_hourly() -> tuple[int, list[datetime]]:
	"""Write satellite-minus-rainfed hourly AET, clamped to nonnegative values."""
	total_written = 0
	missing_timestamps: list[datetime] = []
	rainfed_files = input_year_files()
	with Dataset(SATELLITE_FILE, "r") as satellite_dataset:
		satellite = source_metadata(
			satellite_dataset, SATELLITE_FILE,
			SATELLITE_AET_VARIABLE, SATELLITE_X_VARIABLE, SATELLITE_Y_VARIABLE, SATELLITE_CRS_VARIABLE,
		)
		satellite_var = satellite[0]
		satellite_indices = {timestamp: index for index, timestamp in enumerate(satellite[1])}
		satellite_units = str(getattr(satellite_var, "units", ""))
		if satellite_units != "mm/hr":
			raise ValueError(f"Satellite AET units must be mm/hr, found {satellite_units!r}")

		all_timestamps = [timestamp for year in rainfed_files for timestamp in year_timestamps(year)]
		output_nc, output_irrigation = create_combined_netcdf(
			COMBINED_NETCDF_FILE, all_timestamps,
			satellite[4], len(satellite[2]), len(satellite[3]), satellite[5], satellite_units,
		)
		offset = 0
		try:
			for year, rainfed_path in tqdm(rainfed_files.items(), desc="AET yearly files", unit="year"):
				with Dataset(rainfed_path, "r") as rainfed_dataset:
					rainfed = source_metadata(
						rainfed_dataset, rainfed_path,
						SOURCE_AET_VARIABLE, X_VARIABLE, Y_VARIABLE, CRS_VARIABLE,
					)
					rainfed_var = rainfed[0]
					rainfed_indices = {timestamp: index for index, timestamp in enumerate(rainfed[1])}
					rainfed_units = str(getattr(rainfed_var, "units", ""))
					# Rainfed data is hourly, so mm/dt values equal mm/hr.
					if rainfed_units not in ("mm/hr", "mm/dt"):
						raise ValueError(f"Rainfed AET units must be mm/hr or mm/dt, found {rainfed_units!r} for {year}")

					x_values, y_values, transform, crs = rainfed[2:]
					validate_grids(satellite[2], satellite[3], x_values, y_values)
					rainfed_reverse_y = y_values[0] < y_values[-1]
					height, width = len(y_values), len(x_values)

					timestamps = year_timestamps(year)
					missing_timestamps.extend(
						timestamp for timestamp in timestamps
						if timestamp not in satellite_indices or timestamp not in rainfed_indices
					)
					for index in tqdm(
						range(len(timestamps)), desc=f"AET {year}", unit="hour", leave=False
					):
						timestamp = timestamps[index]
						if timestamp not in satellite_indices or timestamp not in rainfed_indices:
							frame = np.full((height, width), np.nan, dtype=np.float32)
						else:
							satellite_frame = np.ma.filled(
								satellite_var[satellite_indices[timestamp], :, :], np.nan
							).astype(np.float32)
							rainfed_frame = np.ma.filled(
								rainfed_var[rainfed_indices[timestamp], :, :], np.nan
							).astype(np.float32)
							if rainfed_reverse_y:
								rainfed_frame = rainfed_frame[::-1, :]
							frame = np.maximum(satellite_frame - rainfed_frame, 0).astype(np.float32)
						stamp = timestamp.strftime("%Y%m%d%H")
						write_geotiff(
							OUTPUT_DIR / f"{year}_250m" / f"{OUTPUT_PREFIX}_{stamp}.tif",
							frame, transform, crs, satellite_units,
						)
						output_irrigation[offset + index] = np.where(np.isfinite(frame), frame, NODATA)
						total_written += 1
					offset += len(timestamps)
		finally:
			output_nc.close()
	return total_written, missing_timestamps


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written, missing_timestamps = calculate_irrigation_aet_hourly()
	print(f"Wrote {written} hourly irrigation AET GeoTIFFs and combined NetCDF to {OUTPUT_DIR}")
	if missing_timestamps:
		print("Timestamps without matching satellite/rainfed inputs (written as NODATA):")
		print(f"Number of missing timestamps: {len(missing_timestamps)}")
	else:
		print("No unmatched satellite/rainfed timestamps.")