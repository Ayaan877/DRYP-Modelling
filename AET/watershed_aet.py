import fiona
import numpy as np
import rasterio
import re
import warnings
from affine import Affine
from datetime import datetime, timedelta
from pathlib import Path
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_geom
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = "SSEBop"  # "Bhuvan" or "SSEBop"
RESOLUTION_LABEL = "1000m"  # Bhuvan: "5500m" or "750m"; SSEBop: "1000m"

# SSEBop source pixels are 0.009652 deg (~1046 x 1069 m at the watershed), so a 1000 m grid is used
DATASET_CONFIGS = {
	"Bhuvan": {
		"input_root": f"Bhuvan AET {RESOLUTION_LABEL} - Extracted",
		"output_dir": f"Watershed AET {RESOLUTION_LABEL}",
		"file_prefix": "watershed_bhuvan_AET",
		"title": f"Daily Actual Evapotranspiration (AET), Bhuvan {RESOLUTION_LABEL}",
		"units": "mm/day",
		"long_name": "daily actual evapotranspiration",
	},
	"SSEBop": {
		"input_root": "SSEBop Dekadal AET - Extracted",
		"output_dir": "Watershed AET SSEBop",
		"file_prefix": "watershed_ssebop_AET",
		"title": f"Dekadal Actual Evapotranspiration (AET), SSEBop {RESOLUTION_LABEL}",
		"units": "mm/dekad",
		"long_name": "dekadal actual evapotranspiration",
	},
}
if DATASET not in DATASET_CONFIGS:
	raise ValueError(f"DATASET must be one of {', '.join(DATASET_CONFIGS)}")
CONFIG = DATASET_CONFIGS[DATASET]

INPUT_ROOT = SCRIPT_DIR / CONFIG["input_root"]
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
OUTPUT_DIR = SCRIPT_DIR / CONFIG["output_dir"]
SOURCE_CRS = "EPSG:4326"
WATERSHED_CRS = "EPSG:32643"
DATE_PATTERN = re.compile(r"(\d{8})")
SSEBOP_DATE_PATTERN = re.compile(r"^d(\d{4})(0[1-9]|1[0-2])([1-3])_", re.IGNORECASE)

TIME_ORIGIN = "2015-01-01 00:00:00"
TIME_CALENDAR = "gregorian"
NODATA = -99999
MAX_VALID_AET = 65530  # values above this are treated as nodata
GRID_RESOLUTION = int(RESOLUTION_LABEL.removesuffix("m"))  # metres, output grid in WATERSHED_CRS
BUFFER_PIXELS = 3  # extra pixels kept around the watershed footprint

NETCDF_FORMAT = "NETCDF4"
NETCDF_COMPRESSION_LEVEL = 4
TIME_VARIABLE = "time"
LATITUDE_VARIABLE = "lat"
LONGITUDE_VARIABLE = "lon"
AET_VARIABLE = "aet"
CRS_VARIABLE = "crs"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_watershed():
	with fiona.open(WATERSHED_FILE) as src:
		return [feature["geometry"] for feature in src]


def reproject_watershed(watershed_shape):
	return [transform_geom(WATERSHED_CRS, SOURCE_CRS, geom) 
	        for geom in watershed_shape]


def bounds_of(geometries):
	xs, ys = [], []
	for geom in geometries:
		polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
		for polygon in polygons:
			for ring in polygon:
				xs.extend(point[0] for point in ring)
				ys.extend(point[1] for point in ring)
	return min(xs), min(ys), max(xs), max(ys)


def make_target_grid(watershed_shape):
	"""Common grid in WATERSHED_CRS covering the watershed, snapped to GRID_RESOLUTION."""
	xs, ys = [], []
	for geom in watershed_shape:
		rings = geom["coordinates"] if geom["type"] == "Polygon" else [
			ring for polygon in geom["coordinates"] for ring in polygon
		]
		for ring in rings:
			xs.extend(point[0] for point in ring)
			ys.extend(point[1] for point in ring)
	res = GRID_RESOLUTION
	margin = BUFFER_PIXELS * res
	west = np.floor(min(xs) / res) * res - margin
	east = np.ceil(max(xs) / res) * res + margin
	north = np.ceil(max(ys) / res) * res + margin
	south = np.floor(min(ys) / res) * res - margin
	width = int(round((east - west) / res))
	height = int(round((north - south) / res))
	return from_origin(west, north, res, res), width, height


def footprint_mask(watershed_shape, target_transform, width, height):
	"""True for every target pixel within BUFFER_PIXELS of a pixel touching the watershed."""
	touching = ~geometry_mask(
		watershed_shape,
		out_shape=(height, width),
		transform=target_transform,
		all_touched=True,
	)
	padded = np.pad(touching, BUFFER_PIXELS)
	footprint = np.zeros_like(touching)
	size = 2 * BUFFER_PIXELS + 1
	for row_shift in range(size):
		for col_shift in range(size):
			footprint |= padded[row_shift:row_shift + height, col_shift:col_shift + width]
	return footprint


def mask_raster(raster_file, reprojected_watershed, target_transform, width, height, footprint):
	"""Resample (nearest) onto the common UTM grid, keeping every pixel touching the watershed.

	The source is only cropped to a padded bounding box (not masked by the polygon), so
	target pixels that overlap the boundary always sample real source data.
	"""
	with rasterio.open(raster_file) as src:
		if src.crs is None:
			raise RasterioIOError(f"{raster_file.name} has no CRS/georeferencing")
		minx, miny, maxx, maxy = bounds_of(reprojected_watershed)
		pad = 2 * max(abs(src.transform.a), abs(src.transform.e)) + (BUFFER_PIXELS + 1) * GRID_RESOLUTION / 90000
		src_transform = src.transform
		col_start = max(int(np.floor((minx - pad - src_transform.c) / src_transform.a)), 0)
		col_stop = min(int(np.ceil((maxx + pad - src_transform.c) / src_transform.a)), src.width)
		row_start = max(int(np.floor((maxy + pad - src_transform.f) / src_transform.e)), 0)
		row_stop = min(int(np.ceil((miny - pad - src_transform.f) / src_transform.e)), src.height)
		data = src.read(1, window=((row_start, row_stop), (col_start, col_stop))).astype(np.float64)
		transform = src_transform * Affine.translation(col_start, row_start)
		invalid = (data == src.nodata) | (data > MAX_VALID_AET) | ~np.isfinite(data)
		source = np.where(invalid, NODATA, data).astype(np.float32)
		source_crs = src.crs
	target = np.full((height, width), NODATA, dtype=np.float32)
	reproject(
		source,
		target,
		src_transform=transform,
		src_crs=source_crs,
		src_nodata=NODATA,
		dst_transform=target_transform,
		dst_crs=WATERSHED_CRS,
		dst_nodata=NODATA,
		resampling=Resampling.nearest,
	)
	target[~footprint] = NODATA
	return target


def file_date(source_file):
	if DATASET == "SSEBop":
		# dYYYYMMK -> first day of dekad K (1st, 11th or 21st)
		match = SSEBOP_DATE_PATTERN.match(source_file.name)
		if not match:
			raise ValueError(f"No SSEBop dekad found in file name: {source_file.name}")
		year, month, dekad = match.groups()
		return f"{year}{month}{(1, 11, 21)[int(dekad) - 1]:02d}"
	date_match = DATE_PATTERN.search(source_file.stem)
	if not date_match:
		raise ValueError(f"No date found in file name: {source_file.name}")
	return date_match.group(1)


def days_since_origin(date_text):
	origin = datetime.strptime(TIME_ORIGIN, "%Y-%m-%d %H:%M:%S")
	return (datetime.strptime(date_text, "%Y%m%d") - origin).days


def write_year_tifs(year, dates, frames, target_transform, width, height):
	"""Write one GeoTIFF per frame and return the paths written."""
	year_folder = OUTPUT_DIR / f"{year}_{RESOLUTION_LABEL}"
	year_folder.mkdir(parents=True, exist_ok=True)
	written = []
	profile = {
		"driver": "GTiff",
		"dtype": "float32",
		"count": 1,
		"width": width,
		"height": height,
		"crs": CRS.from_string(WATERSHED_CRS),
		"transform": target_transform,
		"nodata": NODATA,
		"compress": "deflate",
	}
	for date_text, frame in zip(dates, frames):
		target_file = year_folder / f"{CONFIG['file_prefix']}_{RESOLUTION_LABEL}_{date_text}.tif"
		with rasterio.open(target_file, "w", **profile) as dst:
			dst.write(frame, 1)
			dst.update_tags(1, units=CONFIG["units"], long_name=CONFIG["long_name"])
		written.append(target_file)
	return written


def write_year_netcdf(year, dates, frames, target_transform, width, height):
	output_file = OUTPUT_DIR / f"{CONFIG['file_prefix']}_{RESOLUTION_LABEL}_{year}.nc"
	order = np.argsort(dates)
	time = np.array([days_since_origin(dates[i]) for i in order], dtype=np.int64)
	if len(np.unique(time)) != len(time):
		raise ValueError(f"Duplicate dates found for {year}")
	crs = CRS.from_string(WATERSHED_CRS)
	x_coordinates = target_transform.c + (np.arange(width) + 0.5) * target_transform.a
	y_coordinates = target_transform.f + (np.arange(height) + 0.5) * target_transform.e

	with Dataset(output_file, "w", format=NETCDF_FORMAT) as output:
		output.Conventions = "CF-1.8"
		output.title = CONFIG["title"]
		output.history = "Created by watershed_aet.py"
		output.createDimension(TIME_VARIABLE, len(time))
		output.createDimension(LATITUDE_VARIABLE, height)
		output.createDimension(LONGITUDE_VARIABLE, width)

		time_variable = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
		lat_variable = output.createVariable(LATITUDE_VARIABLE, "f4", (LATITUDE_VARIABLE,))
		lon_variable = output.createVariable(LONGITUDE_VARIABLE, "f4", (LONGITUDE_VARIABLE,))
		aet_variable = output.createVariable(
			AET_VARIABLE,
			"f4",
			(TIME_VARIABLE, LATITUDE_VARIABLE, LONGITUDE_VARIABLE),
			zlib=True,
			complevel=NETCDF_COMPRESSION_LEVEL,
			chunksizes=(1, height, width),
			fill_value=NODATA,
		)

		time_variable[:] = time
		lat_variable[:] = y_coordinates.astype(np.float32)
		lon_variable[:] = x_coordinates.astype(np.float32)
		aet_variable[:] = np.stack([frames[i] for i in order])

		time_variable.units = f"days since {TIME_ORIGIN}"
		time_variable.calendar = TIME_CALENDAR
		time_variable.standard_name = "time"
		time_variable.long_name = "time"
		time_variable.axis = "T"
		lat_variable.units = "m"
		lon_variable.units = "m"
		lat_variable.long_name = "UTM northing pixel center"
		lon_variable.long_name = "UTM easting pixel center"
		lat_variable.standard_name = "projection_y_coordinate"
		lon_variable.standard_name = "projection_x_coordinate"
		lat_variable.axis = "Y"
		lon_variable.axis = "X"
		aet_variable.long_name = CONFIG["long_name"]
		aet_variable.units = CONFIG["units"]
		aet_variable.coordinates = f"{LATITUDE_VARIABLE} {LONGITUDE_VARIABLE}"
		aet_variable.grid_mapping = CRS_VARIABLE

		crs_variable = output.createVariable(CRS_VARIABLE, "i4")
		crs_variable.long_name = "CRS definition"
		crs_variable.grid_mapping_name = "universal_transverse_mercator"
		crs_variable.epsg_code = WATERSHED_CRS
		crs_variable.spatial_ref = crs.to_wkt()
		crs_variable.crs_wkt = crs.to_wkt()
	return output_file


def list_input_tifs():
	return sorted(
		path for path in INPUT_ROOT.rglob("*.tif")
		if OUTPUT_DIR not in path.parents
	)


def validate_outputs(skipped_files=()):
	"""Compare extracted input tifs (minus intentionally skipped ones) with output frames."""
	skipped = {path for path, _ in skipped_files}
	raw_dates = [file_date(path) for path in list_input_tifs() if path not in skipped]
	out_dates = []
	for nc_path in sorted(OUTPUT_DIR.glob("*.nc")):
		with Dataset(nc_path) as nc:
			origin = datetime.strptime(TIME_ORIGIN, "%Y-%m-%d %H:%M:%S")
			out_dates.extend(
				(origin + timedelta(days=int(day))).strftime("%Y%m%d")
				for day in nc[TIME_VARIABLE][:]
			)
	raw_set, out_set = set(raw_dates), set(out_dates)
	missing = sorted(raw_set - out_set)
	extra = sorted(out_set - raw_set)

	print("\nValidation (extracted input vs. output)")
	print(f"  Input tif files       : {len(raw_dates)} ({len(raw_set)} unique dates)")
	print(f"  Output frames         : {len(out_dates)} ({len(out_set)} unique dates)")
	print(f"  Missing from output   : {len(missing)} {missing[:20]}")
	print(f"  Not in input          : {len(extra)} {extra[:20]}")
	ok = len(raw_dates) == len(out_dates) and not missing and not extra
	print("  RESULT: PASS" if ok else "  RESULT: FAIL - counts/dates do not match")
	return ok

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	watershed_shape = read_watershed()
	reprojected_watershed = reproject_watershed(watershed_shape)

	tif_files = list_input_tifs()

	print(f"Found {len(tif_files)} tif files")
	files_by_year = {}
	for tif_file in tif_files:
		files_by_year.setdefault(file_date(tif_file)[:4], []).append(tif_file)

	target_transform, grid_width, grid_height = make_target_grid(watershed_shape)
	footprint = footprint_mask(watershed_shape, target_transform, grid_width, grid_height)
	skipped_files = []

	for year, year_files in files_by_year.items():
		dates, frames = [], []
		for tif_file in tqdm(year_files, desc=f"AET {year}", unit="file"):
			if tif_file.stat().st_size == 0:
				skipped_files.append((tif_file, "zero-byte file"))
				continue
			try:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", NotGeoreferencedWarning)
					frame = mask_raster(
						tif_file, reprojected_watershed, target_transform, 						grid_width, grid_height, footprint)
			except RasterioIOError as err:
				skipped_files.append((tif_file, str(err)))
				continue
			dates.append(file_date(tif_file))
			frames.append(frame)
		if frames:
			write_year_netcdf(
				year, dates, frames, target_transform, grid_width, grid_height)
			write_year_tifs(
				year, dates, frames, target_transform, grid_width, grid_height)

	print(f"\nExcluded files: {len(skipped_files)}")
	for path, reason in skipped_files:
		print(f"  {path.name}: {reason}")
	print(f"Done. Watershed NetCDFs saved in: {OUTPUT_DIR}")
	validate_outputs(skipped_files)