import fiona
import numpy as np
import rasterio
from datetime import date
from pathlib import Path
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
RESOLUTION_LABEL = "10km"
GRID_RESOLUTION = 10000  # metres, must match RESOLUTION_LABEL
BUFFER_PIXELS = 3  # pixels kept around the watershed footprint
YEARS = range(2015, 2026)
OUTPUT_CRS = "EPSG:32643"
SOURCE_CRS = "EPSG:4326"
SOURCE_NODATA = (-9999, -99999)
NODATA = -99999
TIME_ORIGIN = date(2015, 1, 1)
VARIABLE = "refET"
UNITS = "mm/day"
LONG_NAME = "daily reference evapotranspiration"

# Source name in watershed_data.DATASETS -> output folder and file prefix
OUTPUTS = {
	"ref ET agERA5": {
		"input_root": SCRIPT_DIR / "agERA5 ref ET - Extracted",
		"input_pattern": "agera5_refET_{year}/*.nc",
		"source_variable": "ReferenceET_PenmanMonteith_FAO56",
		"lat_variable": "lat",
		"lon_variable": "lon",
		"output_dir": "Watershed ref ET agERA5 10km",
		"file_prefix": "watershed_agERA5_refET",
		"title": "agERA5 reference ET (FAO-56 Penman-Monteith), Badenhalli subwatershed",
	},
	"ref ET ERA5": {
		"input_root": SCRIPT_DIR / "ERA5 Land ref ET - Extracted",
		"input_pattern": "{year}_daily_pet_Badenhalli_subwatershed.nc",
		"source_variable": "pet",
		"lat_variable": "latitude",
		"lon_variable": "longitude",
		"output_dir": "Watershed ref ET ERA5 Land 10km",
		"file_prefix": "watershed_ERA5Land_refET",
		"title": "ERA5-Land reference ET, Badenhalli subwatershed",
	},
}

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_watershed():
	with fiona.open(WATERSHED_FILE) as src:
		return [feature["geometry"] for feature in src]


def make_target_grid(watershed_shape):
	"""Common grid in OUTPUT_CRS covering the watershed, snapped to GRID_RESOLUTION."""
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


def north_up(values, lat, lon):
	"""Flip the last two axes to north-up, west-left order and return the matching transform."""
	if lat[0] < lat[-1]:
		values, lat = values[..., ::-1, :], lat[::-1]
	if lon[0] > lon[-1]:
		values, lon = values[..., ::-1], lon[::-1]
	res_y = abs(lat[1] - lat[0])
	res_x = abs(lon[1] - lon[0])
	return values, from_origin(lon[0] - res_x / 2, lat[0] + res_y / 2, res_x, res_y)


def source_days(config, year):
	"""Yield (date, 2-D values, source transform) for every day in the year's source files."""
	for path in sorted(config["input_root"].glob(config["input_pattern"].format(year=year))):
		with Dataset(path) as ds:
			time_var = ds["time"]
			moments = num2date(
				time_var[:].astype(np.float64),
				time_var.units,
				calendar=getattr(time_var, "calendar", "standard"),
			)
			if not isinstance(moments, np.ndarray):
				moments = np.array([moments], dtype=object)
			values = np.ma.filled(ds[config["source_variable"]][:].astype(np.float64), np.nan)
			lat = np.asarray(ds[config["lat_variable"]][:], dtype=np.float64)
			lon = np.asarray(ds[config["lon_variable"]][:], dtype=np.float64)
		values, transform = north_up(values, lat, lon)
		for index, moment in enumerate(moments):
			yield date(moment.year, moment.month, moment.day), values[index], transform


def source_lookup(source_transform, target_transform, width, height):
	"""Fractional source (row, col) of each target pixel centre, with integers at source cell centres."""
	cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
	xs, ys = target_transform * (cols, rows)
	coords = warp_transform(OUTPUT_CRS, SOURCE_CRS, xs.ravel().tolist(), ys.ravel().tolist())
	src_cols, src_rows = ~source_transform * (np.asarray(coords[0]), np.asarray(coords[1]))
	return (np.asarray(src_rows) - 0.5).reshape(height, width), (np.asarray(src_cols) - 0.5).reshape(height, width)


def regrid_to_watershed(values, source_transform, target_transform, width, height, footprint):
	"""Bilinear sample of the source under each target pixel centre, using only valid corners."""
	valid = np.isfinite(values)
	for nodata in SOURCE_NODATA:
		valid &= values != nodata
	data = np.where(valid, values, 0.0)
	src_h, src_w = data.shape
	frac_rows, frac_cols = source_lookup(source_transform, target_transform, width, height)
	row0 = np.floor(frac_rows).astype(int)
	col0 = np.floor(frac_cols).astype(int)
	w_row = frac_rows - row0
	w_col = frac_cols - col0
	total = np.zeros((height, width))
	weight = np.zeros((height, width))
	for d_row, d_col in ((0, 0), (0, 1), (1, 0), (1, 1)):
		rr = row0 + d_row
		cc = col0 + d_col
		inside = (rr >= 0) & (rr < src_h) & (cc >= 0) & (cc < src_w)
		rr = np.clip(rr, 0, src_h - 1)
		cc = np.clip(cc, 0, src_w - 1)
		w = (w_row if d_row else 1 - w_row) * (w_col if d_col else 1 - w_col)
		ok = inside & valid[rr, cc]
		total += np.where(ok, w * data[rr, cc], 0.0)
		weight += np.where(ok, w, 0.0)
	target = np.full((height, width), NODATA, dtype=np.float32)
	good = weight > 0
	target[good] = (total[good] / weight[good]).astype(np.float32)
	target[~footprint] = NODATA
	return target


def write_year_netcdf(output_dir, file_prefix, title, year, dates, frames, transform, width, height):
	time_values = np.array([(day - TIME_ORIGIN).days for day in dates], dtype=np.int64)
	x_values = (transform.c + (np.arange(width) + 0.5) * transform.a).astype(np.float32)
	y_values = (transform.f + (np.arange(height) + 0.5) * transform.e).astype(np.float32)
	output_path = output_dir / f"{file_prefix}_{RESOLUTION_LABEL}_{year}.nc"
	with Dataset(output_path, "w", format="NETCDF4") as ds:
		ds.Conventions = "CF-1.8"
		ds.title = title
		ds.history = "Created by watershed_refET.py"
		ds.createDimension("time", len(time_values))
		ds.createDimension("lat", height)
		ds.createDimension("lon", width)

		time_var = ds.createVariable("time", "i8", ("time",))
		time_var.units = f"days since {TIME_ORIGIN.isoformat()} 00:00:00"
		time_var.calendar = "gregorian"
		time_var.standard_name = "time"
		time_var[:] = time_values

		lat_var = ds.createVariable("lat", "f4", ("lat",))
		lat_var.units = "m"
		lat_var.standard_name = "projection_y_coordinate"
		lat_var.long_name = "northing (UTM zone 43N, pixel centre)"
		lat_var[:] = y_values

		lon_var = ds.createVariable("lon", "f4", ("lon",))
		lon_var.units = "m"
		lon_var.standard_name = "projection_x_coordinate"
		lon_var.long_name = "easting (UTM zone 43N, pixel centre)"
		lon_var[:] = x_values

		crs_var = ds.createVariable("crs", "i4")
		crs_var.grid_mapping_name = "transverse_mercator"
		crs_var.latitude_of_projection_origin = 0.0
		crs_var.longitude_of_central_meridian = 75.0
		crs_var.scale_factor_at_central_meridian = 0.9996
		crs_var.false_easting = 500000.0
		crs_var.false_northing = 0.0
		crs_var.epsg_code = OUTPUT_CRS
		crs_var.crs_wkt = CRS.from_string(OUTPUT_CRS).to_wkt()

		data_var = ds.createVariable(
			VARIABLE, "f4", ("time", "lat", "lon"),
			zlib=True, complevel=4, chunksizes=(1, height, width), fill_value=NODATA,
		)
		data_var.units = UNITS
		data_var.long_name = LONG_NAME
		data_var.grid_mapping = "crs"
		data_var[:] = frames


def write_year_tifs(output_dir, file_prefix, year, dates, frames, transform):
	year_dir = output_dir / f"{year}_{RESOLUTION_LABEL}"
	year_dir.mkdir(parents=True, exist_ok=True)
	profile = {
		"driver": "GTiff",
		"dtype": "float32",
		"count": 1,
		"height": frames.shape[1],
		"width": frames.shape[2],
		"crs": CRS.from_string(OUTPUT_CRS),
		"transform": transform,
		"nodata": NODATA,
		"compress": "deflate",
	}
	for day, frame in zip(dates, frames):
		output_path = year_dir / f"{file_prefix}_{RESOLUTION_LABEL}_{day.strftime('%Y%m%d')}.tif"
		with rasterio.open(output_path, "w", **profile) as dst:
			dst.write(frame, 1)
			dst.update_tags(1, units=UNITS, long_name=LONG_NAME)


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	watershed_shape = read_watershed()
	target_transform, width, height = make_target_grid(watershed_shape)
	footprint = footprint_mask(watershed_shape, target_transform, width, height)
	for source_name, config in OUTPUTS.items():
		output_dir = SCRIPT_DIR / config["output_dir"]
		output_dir.mkdir(parents=True, exist_ok=True)
		for year in YEARS:
			days = sorted(source_days(config, year), key=lambda item: item[0])
			if not days:
				print(f"No source files for {source_name} {year}, skipped")
				continue
			dates = [day for day, _, _ in days]
			frames = np.stack([
				regrid_to_watershed(values, transform, target_transform, width, height, footprint)
				for _, values, transform in tqdm(days, desc=f"{source_name} {year}", unit="day")
			])
			write_year_netcdf(output_dir, config["file_prefix"], config["title"], year, dates, frames, target_transform, width, height)
			write_year_tifs(output_dir, config["file_prefix"], year, dates, frames, target_transform)
		print(f"Wrote {source_name} to {output_dir}")
