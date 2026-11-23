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
VERSION = "v2"
INPUT_NETCDF_FILE = SCRIPT_DIR / f"Watershed irrigation AET hourly 250m {VERSION}" / "irrigation_AET_hourly_250m_all_years.nc"
OUTPUT_DIR = SCRIPT_DIR / f"Watershed gw abstraction hourly 250m {VERSION}"
OUTPUT_PREFIX = "gw_abstraction_hourly_250m"
COMBINED_NETCDF_FILE = OUTPUT_DIR / f"{OUTPUT_PREFIX}_all_years.nc"

IRRIGATION_VARIABLE = "irr"
GW_ABSTRACTION_VARIABLE = "gw_abs"
TIME_VARIABLE = "time"
X_VARIABLE = "x"
Y_VARIABLE = "y"
CRS_VARIABLE = "crs"
DEFAULT_CRS = "EPSG:32643"
NODATA = -9999.0
NETCDF_COMPRESSION_LEVEL = 4
PIXEL_SIZE_M = 250
MM_TO_M = 0.001
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2
ABSTRACTION_FACTOR = 1.0 / 0.85
OUTPUT_UNITS = "m3/hr"

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


def source_metadata(
	dataset: Dataset, path: Path,
) -> tuple[Any, Any, np.ndarray, np.ndarray, rasterio.Affine, CRS, list[datetime]]:
	for name in (IRRIGATION_VARIABLE, TIME_VARIABLE, X_VARIABLE, Y_VARIABLE):
		if name not in dataset.variables:
			raise ValueError(f"Configured variable {name!r} was not found in {path}")

	irrigation_variable = dataset.variables[IRRIGATION_VARIABLE]
	time_variable = dataset.variables[TIME_VARIABLE]
	x_variable = dataset.variables[X_VARIABLE]
	y_variable = dataset.variables[Y_VARIABLE]
	if irrigation_variable.dimensions != (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE):
		raise ValueError(f"{IRRIGATION_VARIABLE!r} must use dimensions (time, y, x) in {path}")
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
	if len(timestamps) != irrigation_variable.shape[0]:
		raise ValueError(f"AET time/data length mismatch in {path}")
	return irrigation_variable, time_variable, x_values, y_values, transform, crs, timestamps


def create_combined_netcdf(
	path: Path,
	time_template: Any,
	time_values: np.ndarray,
	input_x: np.ndarray,
	input_y: np.ndarray,
	crs: CRS,
	units: str,
) -> tuple[Dataset, Any]:
	path.parent.mkdir(parents=True, exist_ok=True)
	height, width = len(input_y), len(input_x)
	output = Dataset(path, "w", format="NETCDF4")
	output.Conventions = "CF-1.8"
	output.title = "Hourly gw abstraction"
	output.createDimension(TIME_VARIABLE, len(time_values))
	output.createDimension(Y_VARIABLE, height)
	output.createDimension(X_VARIABLE, width)

	time_var = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
	time_var[:] = time_values
	time_var.units = time_template.units
	time_var.calendar = getattr(time_template, "calendar", "standard")
	time_var.standard_name = getattr(time_template, "standard_name", "time")
	x_var = output.createVariable(X_VARIABLE, "f4", (X_VARIABLE,))
	y_var = output.createVariable(Y_VARIABLE, "f4", (Y_VARIABLE,))
	x_var[:] = input_x
	y_var[:] = input_y
	x_var.axis = "X"
	y_var.axis = "Y"
	x_var.standard_name = "projection_x_coordinate"
	y_var.standard_name = "projection_y_coordinate"
	x_var.units = "m"
	y_var.units = "m"
	crs_var = output.createVariable(CRS_VARIABLE, "i4")
	crs_var.spatial_ref = crs.to_wkt()
	crs_var.crs_wkt = crs.to_wkt()
	abstraction_var = output.createVariable(
		GW_ABSTRACTION_VARIABLE, "f4", (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE), zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL, chunksizes=(1, height, width),
		fill_value=NODATA,
	)
	abstraction_var.units = units
	abstraction_var.long_name = "Hourly gw abstraction"
	abstraction_var.grid_mapping = CRS_VARIABLE
	return output, abstraction_var


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
			long_name="Hourly gw abstraction",
		)


def negative_abstraction_frame(frame: np.ndarray) -> np.ndarray:
	values = np.asarray(frame, dtype=np.float32).copy()
	finite = np.isfinite(values)
	values[finite] = -np.abs(values[finite])
	return values


def fix_existing_outputs_negative() -> tuple[int, int, int]:
	"""Make every finite nonzero value negative in existing GW output files."""
	raster_count = 0
	positive_count = 0
	if not COMBINED_NETCDF_FILE.is_file():
		raise FileNotFoundError(f"Combined GW abstraction NetCDF was not found: {COMBINED_NETCDF_FILE}")

	with Dataset(COMBINED_NETCDF_FILE, "r+") as dataset:
		if GW_ABSTRACTION_VARIABLE not in dataset.variables:
			raise ValueError(f"{GW_ABSTRACTION_VARIABLE!r} was not found in {COMBINED_NETCDF_FILE}")
		abstraction = dataset.variables[GW_ABSTRACTION_VARIABLE]
		for index in tqdm(range(abstraction.shape[0]), desc="Correcting GW NetCDF", unit="hour"):
			frame = np.ma.filled(abstraction[index, :, :], np.nan).astype(np.float32)
			positive = np.isfinite(frame) & (frame > 0)
			positive_count += int(np.count_nonzero(positive))
			frame = negative_abstraction_frame(frame)
			abstraction[index, :, :] = np.where(np.isfinite(frame), frame, NODATA)

	raster_paths = sorted(OUTPUT_DIR.glob(f"*_250m/{OUTPUT_PREFIX}_*.tif"))
	for raster_path in tqdm(raster_paths, desc="Correcting GW rasters", unit="raster"):
		with rasterio.open(raster_path, "r+") as raster:
			frame = np.ma.filled(raster.read(1, masked=True), np.nan).astype(np.float32)
			positive = np.isfinite(frame) & (frame > 0)
			positive_count += int(np.count_nonzero(positive))
			frame = negative_abstraction_frame(frame)
			nodata = raster.nodata if raster.nodata is not None else NODATA
			raster.write(np.where(np.isfinite(frame), frame, nodata).astype(np.float32), 1)
		raster_count += 1
	return 1, raster_count, positive_count


def calculate_gw_abstraction_hourly() -> int:
	"""Scale combined irrigation AET into hourly gw abstraction outputs."""
	total_written = 0
	with Dataset(INPUT_NETCDF_FILE, "r") as source:
		irrigation_variable, time_variable, x_values, y_values, transform, crs, timestamps = source_metadata(
			source, INPUT_NETCDF_FILE
		)
		input_units = str(getattr(irrigation_variable, "units", "mm/hr"))
		if input_units != "mm/hr":
			raise ValueError(f"Expected irrigation units of 'mm/hr', found {input_units!r}")
		reverse_y = y_values[0] < y_values[-1]
		output_nc, output_abstraction = create_combined_netcdf(
			COMBINED_NETCDF_FILE, time_variable, np.asarray(time_variable[:]),
			x_values, y_values, crs, OUTPUT_UNITS,
		)
		try:
			for index, timestamp in enumerate(tqdm(timestamps, desc="Abstraction hours", unit="hour")):
				frame = np.ma.filled(irrigation_variable[index, :, :], np.nan).astype(np.float32)
				frame *= ABSTRACTION_FACTOR * PIXEL_AREA_M2 * MM_TO_M
				frame = negative_abstraction_frame(frame)
				output_abstraction[index] = np.where(np.isfinite(frame), frame, NODATA)
				geotiff_frame = frame[::-1, :] if reverse_y else frame
				stamp = timestamp.strftime("%Y%m%d%H")
				write_geotiff(
					OUTPUT_DIR / f"{timestamp.year}_250m" / f"{OUTPUT_PREFIX}_{stamp}.tif",
					geotiff_frame, transform, crs, OUTPUT_UNITS,
				)
				total_written += 1
		finally:
			output_nc.close()
	return total_written


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written = calculate_gw_abstraction_hourly()
	print(
		f"Wrote {written} hourly gw abstraction GeoTIFFs and combined NetCDF "
		f"to {OUTPUT_DIR}"
	)