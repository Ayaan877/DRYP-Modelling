import re
from math import ceil
from datetime import date
from pathlib import Path

import fiona
import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import bounds, geometry_mask
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform_geom
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR = 2018
YEAR_FOLDER = SCRIPT_DIR / str(YEAR)
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
OUTPUT_FILE = SCRIPT_DIR / f"AET_{YEAR}.nc"
ALLOW_MISSING = True

INPUT_PATTERN = re.compile(r"AET_SNPP_N_EB_DY_(\d{8})\.tif$")
SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:32643"
TARGET_RESOLUTION = 1000.0
TIME_ORIGIN = f"{YEAR}-01-01 00:00:00"
TIME_CALENDAR = "gregorian"

NODATA = -99999.0
SOURCE_NODATA_THRESHOLD = 65530.0

NETCDF_FORMAT = "NETCDF4"
NETCDF_COMPRESSION_LEVEL = 4
NETCDF_CHUNK_SIZE = 256

TIME_VARIABLE = "time"
TIME_BOUNDS_VARIABLE = "time_bounds"
LATITUDE_VARIABLE = "lat"
LONGITUDE_VARIABLE = "lon"
AET_VARIABLE = "aet"
CRS_VARIABLE = "crs"


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_daily_files(year_folder, year):
	files_by_date = {}
	for raster_file in year_folder.glob("*.tif"):
		match = INPUT_PATTERN.fullmatch(raster_file.name)
		if match is None:
			continue

		file_date = date(
			int(match.group(1)[:4]),
			int(match.group(1)[4:6]),
			int(match.group(1)[6:]),
		)
		if file_date.year != year:
			continue
		if file_date in files_by_date:
			raise ValueError(f"Multiple GeoTIFFs found for {file_date}")
		files_by_date[file_date] = raster_file

	if not files_by_date:
		raise FileNotFoundError(f"No daily AET GeoTIFFs found in {year_folder}")
	return files_by_date


def get_expected_dates(year):
	start = date(year, 1, 1)
	number_of_days = (date(year + 1, 1, 1) - start).days
	return [date.fromordinal(start.toordinal() + day) for day in range(number_of_days)]


def read_watershed_geometries(watershed_file, target_crs):
	with fiona.open(watershed_file) as watershed:
		source_crs = watershed.crs_wkt or watershed.crs
		if not source_crs:
			raise ValueError(f"Watershed has no CRS information: {watershed_file}")
		geometries = [
			transform_geom(source_crs, target_crs, feature["geometry"])
			for feature in watershed
		]

	if not geometries:
		raise ValueError(f"Watershed contains no geometries: {watershed_file}")
	return geometries


def get_crop_window(source, geometries):
	geometry_bounds = [bounds(geometry) for geometry in geometries]
	x_min = min(geometry_bound[0] for geometry_bound in geometry_bounds)
	y_min = min(geometry_bound[1] for geometry_bound in geometry_bounds)
	x_max = max(geometry_bound[2] for geometry_bound in geometry_bounds)
	y_max = max(geometry_bound[3] for geometry_bound in geometry_bounds)

	window = from_bounds(
		x_min, y_min, x_max, y_max, transform=source.transform
	).round_offsets().round_lengths()
	window = Window.from_slices(
		slice(max(0, int(window.row_off)), min(source.height, int(window.row_off + window.height))),
		slice(max(0, int(window.col_off)), min(source.width, int(window.col_off + window.width))),
	)
	if window.width <= 0 or window.height <= 0:
		raise ValueError("Watershed does not overlap the AET raster")
	return window


def get_target_grid(geometries):
	geometry_bounds = [bounds(geometry) for geometry in geometries]
	x_min = min(geometry_bound[0] for geometry_bound in geometry_bounds)
	y_min = min(geometry_bound[1] for geometry_bound in geometry_bounds)
	x_max = max(geometry_bound[2] for geometry_bound in geometry_bounds)
	y_max = max(geometry_bound[3] for geometry_bound in geometry_bounds)
	width = ceil((x_max - x_min) / TARGET_RESOLUTION)
	height = ceil((y_max - y_min) / TARGET_RESOLUTION)
	transform = from_origin(
		x_min,
		y_min + height * TARGET_RESOLUTION,
		TARGET_RESOLUTION,
		TARGET_RESOLUTION,
	)
	return transform, width, height


def get_pixel_coordinates(transform, width, height):
	left, bottom, right, top = array_bounds(height, width, transform)
	x_coordinates = np.linspace(
		left + (right - left) / (2 * width),
		right - (right - left) / (2 * width),
		width,
	)
	y_coordinates = np.linspace(
		top - (top - bottom) / (2 * height),
		bottom + (top - bottom) / (2 * height),
		height,
	)
	return x_coordinates, y_coordinates


def reproject_aet(
	source,
	window,
	target_transform,
	target_width,
	target_height,
	watershed_mask,
):
	source_data = source.read(1, window=window).astype(np.float32)
	source_data[source_data >= SOURCE_NODATA_THRESHOLD] = NODATA
	target_data = np.full(
		(target_height, target_width),
		NODATA,
		dtype=np.float32,
	)
	reproject(
		source=source_data,
		destination=target_data,
		src_transform=source.window_transform(window),
		src_crs=source.crs,
		dst_transform=target_transform,
		dst_crs=TARGET_CRS,
		resampling=Resampling.nearest,
		src_nodata=NODATA,
		dst_nodata=NODATA,
	)
	target_data[~watershed_mask] = NODATA
	return target_data


def create_netcdf_variables(output, time, x_coordinates, y_coordinates):
	height = len(y_coordinates)
	width = len(x_coordinates)
	output.Conventions = "CF-1.8"
	output.title = "Daily actual evapotranspiration in the watershed"
	output.history = "Created by make_aet_year.py"
	output.source_units = "mm/day"
	output.output_units = "mm/day"
	output.createDimension(TIME_VARIABLE, len(time))
	output.createDimension(LATITUDE_VARIABLE, height)
	output.createDimension(LONGITUDE_VARIABLE, width)
	output.createDimension("nv", 2)

	time_variable = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
	time_bounds = output.createVariable(
		TIME_BOUNDS_VARIABLE, "i8", (TIME_VARIABLE, "nv")
	)
	lat_variable = output.createVariable(LATITUDE_VARIABLE, "f4", (LATITUDE_VARIABLE,))
	lon_variable = output.createVariable(LONGITUDE_VARIABLE, "f4", (LONGITUDE_VARIABLE,))
	aet_variable = output.createVariable(
		AET_VARIABLE,
		"f4",
		(TIME_VARIABLE, LATITUDE_VARIABLE, LONGITUDE_VARIABLE),
		zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL,
		chunksizes=(1, min(NETCDF_CHUNK_SIZE, height), min(NETCDF_CHUNK_SIZE, width)),
		fill_value=NODATA,
	)

	time_variable[:] = time
	time_bounds[:, 0] = time
	time_bounds[:, 1] = time + 1
	lat_variable[:] = y_coordinates.astype(np.float32)
	lon_variable[:] = x_coordinates.astype(np.float32)
	return time_variable, time_bounds, lat_variable, lon_variable, aet_variable


def add_netcdf_metadata(output, variables, crs):
	time_variable, time_bounds, lat_variable, lon_variable, aet_variable = variables
	time_variable.units = f"days since {TIME_ORIGIN}"
	time_variable.calendar = TIME_CALENDAR
	time_variable.standard_name = "time"
	time_variable.long_name = "time"
	time_variable.axis = "T"
	time_variable.bounds = TIME_BOUNDS_VARIABLE
	time_bounds.units = time_variable.units
	time_bounds.calendar = TIME_CALENDAR

	lat_variable.units = "m"
	lat_variable.long_name = "UTM northing pixel center"
	lat_variable.standard_name = "projection_y_coordinate"
	lat_variable.axis = "Y"
	lon_variable.units = "m"
	lon_variable.long_name = "UTM easting pixel center"
	lon_variable.standard_name = "projection_x_coordinate"
	lon_variable.axis = "X"

	aet_variable.long_name = "daily actual evapotranspiration"
	aet_variable.units = "mm day-1"
	aet_variable.coordinates = f"{LATITUDE_VARIABLE} {LONGITUDE_VARIABLE}"
	aet_variable.grid_mapping = CRS_VARIABLE
	aet_variable.valid_min = 0.0
	aet_variable.source_nodata_values = "65530-65535"

	crs_variable = output.createVariable(CRS_VARIABLE, "i4")
	crs_variable.long_name = "CRS definition"
	crs_variable.grid_mapping_name = "universal_transverse_mercator"
	crs_variable.epsg_code = TARGET_CRS
	crs_variable.spatial_ref = crs.to_wkt()
	crs_variable.crs_wkt = crs.to_wkt()


def write_aet_netcdf(
	output_file,
	dates,
	files_by_date,
	watershed_file,
	allow_missing,
	):
	missing_dates = [file_date for file_date in dates if file_date not in files_by_date]
	if missing_dates and not allow_missing:
		missing = ", ".join(file_date.isoformat() for file_date in missing_dates)
		raise ValueError(f"Missing dates: {missing}")

	first_file = files_by_date[min(files_by_date)]
	with rasterio.open(first_file) as source:
		if source.crs is None or source.crs.to_string() != SOURCE_CRS:
			raise ValueError(f"Expected {SOURCE_CRS}; found {source.crs}")
		source_geometries = read_watershed_geometries(
			watershed_file, source.crs
		)
		target_geometries = read_watershed_geometries(
			watershed_file, TARGET_CRS
		)
		window = get_crop_window(source, source_geometries)
		target_transform, target_width, target_height = get_target_grid(
			target_geometries
		)
		watershed_mask = geometry_mask(
			target_geometries,
			out_shape=(target_height, target_width),
			transform=target_transform,
			invert=True,
			all_touched=True,
		)
		x_coordinates, y_coordinates = get_pixel_coordinates(
			target_transform, target_width, target_height
		)
		grid = (source.crs, source.transform, source.width, source.height)
		target_crs = CRS.from_user_input(TARGET_CRS)

	output_file.parent.mkdir(parents=True, exist_ok=True)
	with Dataset(output_file, "w", format=NETCDF_FORMAT) as output:
		time = np.arange(len(dates), dtype=np.int64)
		variables = create_netcdf_variables(
			output, time, x_coordinates, y_coordinates
		)
		add_netcdf_metadata(output, variables, target_crs)
		aet_variable = variables[-1]

		for index, file_date in tqdm(
			enumerate(dates),
			total=len(dates),
			desc="Writing AET rasters",
			unit="day",
		):
			if file_date not in files_by_date:
				aet_variable[index, :, :] = NODATA
				continue
			with rasterio.open(files_by_date[file_date]) as source:
				current_grid = (
					source.crs, source.transform, source.width, source.height
				)
				if current_grid != grid:
					raise ValueError(f"Raster grid differs: {source.name}")
				aet_variable[index, :, :] = reproject_aet(
					source,
					window,
					target_transform,
					target_width,
					target_height,
					watershed_mask,
				)

	print(f"Wrote {output_file}")
	print(f"Grid: {target_width} columns x {target_height} rows")
	print(f"Dates: {len(dates)} ({len(missing_dates)} missing)")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def make_aet_year():
	files_by_date = read_daily_files(YEAR_FOLDER, YEAR)
	dates = get_expected_dates(YEAR)
	write_aet_netcdf(
		OUTPUT_FILE,
		dates,
		files_by_date,
		WATERSHED_FILE,
		ALLOW_MISSING,
	)


if __name__ == "__main__":
	make_aet_year()
