from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SATELLITE_INPUT_DIR = SCRIPT_DIR / "Watershed AET hourly 250m"
SATELLITE_PREFIX = "watershed_AET_hourly_250m"
RAINFED_INPUT_DIR = SCRIPT_DIR / "Watershed_rainfed_AET_hourly_250m"
RAINFED_PREFIX = "rainfed_AET_hourly"

OUTPUT_DIR = SCRIPT_DIR / "Watershed irrigation AET hourly 250m"
OUTPUT_PREFIX = "irrigation_AET_hourly_250m"

SOURCE_AET_VARIABLE = "aet"
IRRIGATION_VARIABLE = "irr"
TIME_VARIABLE = "time"
X_VARIABLE = "x"
Y_VARIABLE = "y"
CRS_VARIABLE = "crs"
DEFAULT_CRS = "EPSG:32643"
NODATA = -99999.0
NETCDF_COMPRESSION_LEVEL = 4
TIME_ORIGIN = datetime(2015, 1, 1)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def input_year_files() -> dict[int, tuple[Path, Path]]:
	"""Pair yearly satellite and rainfed NetCDF inputs by year."""
	satellite_files: dict[int, Path] = {}
	for path in sorted(SATELLITE_INPUT_DIR.glob(f"{SATELLITE_PREFIX}_*.nc")):
		try:
			year = int(path.stem.rsplit("_", 1)[1])
		except (IndexError, ValueError) as error:
			raise ValueError(f"Could not determine year from {path.name}") from error
		if year in satellite_files:
			raise ValueError(f"Duplicate satellite NetCDF for {year}: {path}")
		satellite_files[year] = path
	if not satellite_files:
		raise FileNotFoundError(f"No satellite yearly NetCDFs found in {SATELLITE_INPUT_DIR}")

	pairs: dict[int, tuple[Path, Path]] = {}
	for year, satellite_path in sorted(satellite_files.items()):
		rainfed_path = RAINFED_INPUT_DIR / f"{RAINFED_PREFIX}_{year}.nc"
		if not rainfed_path.is_file():
			raise FileNotFoundError(f"Rainfed NetCDF for {year} was not found: {rainfed_path}")
		pairs[year] = satellite_path, rainfed_path
	return pairs


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
) -> tuple[Any, list[datetime], np.ndarray, np.ndarray, rasterio.Affine, CRS]:
	for name in (SOURCE_AET_VARIABLE, TIME_VARIABLE, X_VARIABLE, Y_VARIABLE):
		if name not in dataset.variables:
			raise ValueError(f"Configured variable {name!r} was not found in {path}")

	aet_variable = dataset.variables[SOURCE_AET_VARIABLE]
	time_variable = dataset.variables[TIME_VARIABLE]
	x_variable = dataset.variables[X_VARIABLE]
	y_variable = dataset.variables[Y_VARIABLE]
	if aet_variable.dimensions != (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE):
		raise ValueError(f"{SOURCE_AET_VARIABLE!r} must use dimensions (time, y, x) in {path}")
	if x_variable.dimensions != (X_VARIABLE,) or y_variable.dimensions != (Y_VARIABLE,):
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
	crs_variable = dataset.variables.get(CRS_VARIABLE)
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


def align_frame(
	frame: np.ndarray,
	source_transform: rasterio.Affine,
	source_crs: CRS,
	target_shape: tuple[int, int],
	target_transform: rasterio.Affine,
	target_crs: CRS,
) -> np.ndarray:
	"""Resample one satellite frame onto the rainfed grid."""
	aligned = np.full(target_shape, np.nan, dtype=np.float32)
	reproject(
		frame,
		aligned,
		src_transform=source_transform,
		src_crs=source_crs,
		src_nodata=np.nan,
		dst_transform=target_transform,
		dst_crs=target_crs,
		dst_nodata=np.nan,
		resampling=Resampling.bilinear,
	)
	return aligned


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


def calculate_irrigation_aet_hourly() -> tuple[int, list[datetime]]:
	"""Write satellite-minus-rainfed hourly AET, clamped to nonnegative values."""
	total_written = 0
	missing_timestamps: list[datetime] = []
	for year, (satellite_path, rainfed_path) in tqdm(
		input_year_files().items(), desc="AET yearly files", unit="year"
	):
		with Dataset(satellite_path, "r") as satellite_dataset, Dataset(rainfed_path, "r") as rainfed_dataset:
			satellite = source_metadata(satellite_dataset, satellite_path)
			rainfed = source_metadata(rainfed_dataset, rainfed_path)
			satellite_var = satellite[0]
			rainfed_var = rainfed[0]
			satellite_indices = {timestamp: index for index, timestamp in enumerate(satellite[1])}
			rainfed_indices = {timestamp: index for index, timestamp in enumerate(rainfed[1])}
			start = datetime(year, 1, 1)
			next_year = datetime(year + 1, 1, 1)
			timestamps = [
				start + timedelta(hours=hour)
				for hour in range(int((next_year - start).total_seconds() // 3600))
			]
			missing_timestamps.extend(
				timestamp for timestamp in timestamps
				if timestamp not in satellite_indices or timestamp not in rainfed_indices
			)
			x_values, y_values, transform, crs = rainfed[2:]
			satellite_reverse_y = satellite[3][0] < satellite[3][-1]
			rainfed_reverse_y = y_values[0] < y_values[-1]
			units = str(getattr(satellite_var, "units", "mm/hr"))
			rainfed_units = str(getattr(rainfed_var, "units", units))
			if not (units == "mm/hr" and rainfed_units == "mm/dt"):
				raise ValueError(f"Satellite and rainfed AET units do not match for {year}")
			height, width = len(y_values), len(x_values)
			output_nc, output_irrigation = create_year_netcdf(
				OUTPUT_DIR / f"{OUTPUT_PREFIX}_{year}.nc",
				timestamps, transform, width, height, crs, units,
			)
			try:
				for index in tqdm(
					range(len(timestamps)), desc=f"AET {year}", unit="hour", leave=False
				):
					timestamp = timestamps[index]
					if timestamp not in satellite_indices or timestamp not in rainfed_indices:
						frame = np.zeros((height, width), dtype=np.float32)
					else:
						satellite_frame = np.ma.filled(
							satellite_var[satellite_indices[timestamp], :, :], np.nan
						).astype(np.float32)
						if satellite_reverse_y:
							satellite_frame = satellite_frame[::-1, :]
						satellite_frame = align_frame(
							satellite_frame, satellite[4], satellite[5],
							(height, width), transform, crs,
						)
						rainfed_frame = np.ma.filled(
							rainfed_var[rainfed_indices[timestamp], :, :], np.nan
						).astype(np.float32)
						if rainfed_reverse_y:
							rainfed_frame = rainfed_frame[::-1, :]
						valid = np.isfinite(satellite_frame) & np.isfinite(rainfed_frame)
						frame = np.full(satellite_frame.shape, np.nan, dtype=np.float32)
						frame[valid] = np.maximum(satellite_frame[valid] - rainfed_frame[valid], 0)
					stamp = timestamp.strftime("%Y%m%d%H")
					write_geotiff(
						OUTPUT_DIR / f"{year}_250m" / f"{OUTPUT_PREFIX}_{stamp}.tif",
						frame, transform, crs, units,
					)
					output_irrigation[index] = np.where(np.isfinite(frame), frame, NODATA)
					total_written += 1
			finally:
				output_nc.close()
	return total_written, missing_timestamps


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written, missing_timestamps = calculate_irrigation_aet_hourly()
	print(f"Wrote {written} hourly irrigation AET GeoTIFFs and yearly NetCDFs to {OUTPUT_DIR}")
	if missing_timestamps:
		print("Timestamps without matching satellite/rainfed inputs (written as zero):")
		for timestamp in missing_timestamps:
			print(timestamp.strftime("%Y-%m-%d %H:%M"))
	else:
		print("No unmatched satellite/rainfed timestamps.")