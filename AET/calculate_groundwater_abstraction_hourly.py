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
INPUT_DIR = SCRIPT_DIR / "Watershed irrigation AET hourly 250m"
INPUT_PREFIX = "irrigation_AET_hourly_250m"
OUTPUT_DIR = SCRIPT_DIR / "Watershed gw abstraction hourly 250m"
OUTPUT_PREFIX = "gw_abstraction_hourly_250m"

IRRIGATION_VARIABLE = "irr"
GW_ABSTRACTION_VARIABLE = "gw_abs"
TIME_VARIABLE = "time"
X_VARIABLE = "x"
Y_VARIABLE = "y"
CRS_VARIABLE = "crs"
DEFAULT_CRS = "EPSG:32643"
NODATA = -99999.0
NETCDF_COMPRESSION_LEVEL = 4
PIXEL_SIZE_M = 250
MM_TO_M = 0.001
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2
ABSTRACTION_FACTOR = 1.0 / 0.85
OUTPUT_UNITS = "m3/hr"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def input_year_files() -> dict[int, Path]:
	"""Index irrigation NetCDF files by year."""
	files: dict[int, Path] = {}
	for path in sorted(INPUT_DIR.glob(f"{INPUT_PREFIX}_*.nc")):
		try:
			year = int(path.stem.rsplit("_", 1)[1])
		except (IndexError, ValueError) as error:
			raise ValueError(f"Could not determine year from {path.name}") from error
		if year in files:
			raise ValueError(f"Duplicate irrigation NetCDF for {year}: {path}")
		files[year] = path
	if not files:
		raise FileNotFoundError(f"No yearly irrigation NetCDFs found in {INPUT_DIR}")
	return files


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


def create_year_netcdf(
	path: Path,
	input_time: Any,
	input_x: np.ndarray,
	input_y: np.ndarray,
	transform: rasterio.Affine,
	crs: CRS,
	units: str,
) -> tuple[Dataset, Any]:
	path.parent.mkdir(parents=True, exist_ok=True)
	height, width = len(input_y), len(input_x)
	output = Dataset(path, "w", format="NETCDF4")
	output.Conventions = "CF-1.8"
	output.title = "Hourly gw abstraction"
	output.createDimension(TIME_VARIABLE, len(input_time))
	output.createDimension(Y_VARIABLE, height)
	output.createDimension(X_VARIABLE, width)

	time_var = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
	time_var[:] = input_time[:]
	time_var.units = input_time.units
	time_var.calendar = getattr(input_time, "calendar", "standard")
	time_var.standard_name = getattr(input_time, "standard_name", "time")
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


def calculate_gw_abstraction_hourly() -> int:
	"""Scale irrigation AET into hourly gw abstraction outputs."""
	total_written = 0
	for year, input_path in tqdm(
		input_year_files().items(), desc="Abstraction yearly files", unit="year"
	):
		with Dataset(input_path, "r") as source:
			irrigation_variable, time_variable, x_values, y_values, transform, crs, timestamps = source_metadata(
				source, input_path
			)
			input_units = str(getattr(irrigation_variable, "units", "mm/hr"))
			if input_units != "mm/hr":
				raise ValueError(f"Expected irrigation units of 'mm/hr', found {input_units!r} in {input_path}")
			units = OUTPUT_UNITS
			height, width = len(y_values), len(x_values)
			reverse_y = y_values[0] < y_values[-1]
			output_nc, output_abstraction = create_year_netcdf(
				OUTPUT_DIR / f"{OUTPUT_PREFIX}_{year}.nc",
				time_variable, x_values, y_values, transform, crs, units,
			)
			try:
				for index, timestamp in enumerate(tqdm(
					timestamps, desc=f"Abstraction {year}", unit="hour", leave=False
				)):
					frame = np.ma.filled(irrigation_variable[index, :, :], np.nan).astype(np.float32)
					frame[~np.isfinite(frame)] = np.nan
					frame *= ABSTRACTION_FACTOR * PIXEL_AREA_M2 * MM_TO_M
					output_abstraction[index] = np.where(np.isfinite(frame), frame, NODATA)
					geotiff_frame = frame[::-1, :] if reverse_y else frame
					stamp = timestamp.strftime("%Y%m%d%H")
					write_geotiff(
						OUTPUT_DIR / f"{year}_250m" / f"{OUTPUT_PREFIX}_{stamp}.tif",
						geotiff_frame, transform, crs, units,
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
		f"Wrote {written} hourly gw abstraction GeoTIFFs and yearly NetCDFs "
		f"to {OUTPUT_DIR}"
	)