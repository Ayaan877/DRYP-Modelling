import re
import numpy as np
import rasterio
from datetime import date, timedelta
from pathlib import Path
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESOLUTIONS = ("750m", "5500m")
PET_DIR = SCRIPT_DIR / "Daily PET"
PET_FILE_PATTERN = "{year}_daily_pet_Badenhalli_subwatershed.nc"
PET_VARIABLE = "pet"
PET_TIME_VARIABLE = "time"
PET_LAT_VARIABLE = "latitude"
PET_LON_VARIABLE = "longitude"
PET_CRS = "EPSG:4326"
OUTPUT_CRS = "EPSG:32643"
NODATA = -99999
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def aet_files_by_year(resolution):
	aet_dir = SCRIPT_DIR / f"Watershed AET {resolution}"
	files = {}
	for path in sorted(aet_dir.rglob(f"watershed_bhuvan_AET_{resolution}_*.tif")):
		match = DATE_PATTERN.search(path.name)
		if match:
			files.setdefault(int(match.group(1)[:4]), []).append((match.group(1), path))
	return files


def load_pet(year):
	"""PET cube (time, lat, lon), date -> time index, and the lat/lon cell centres."""
	with Dataset(PET_DIR / PET_FILE_PATTERN.format(year=year)) as ds:
		units = ds[PET_TIME_VARIABLE].units  # "days since YYYY-MM-DD"
		origin = date.fromisoformat(units.split("since")[1].split()[0])
		days = np.ma.filled(ds[PET_TIME_VARIABLE][:], -1).astype(int)
		dates = [(origin + timedelta(days=int(d))).strftime("%Y%m%d") for d in days]
		pet = np.ma.filled(ds[PET_VARIABLE][:].astype(np.float64), np.nan)
		lat = np.asarray(ds[PET_LAT_VARIABLE][:], dtype=np.float64)
		lon = np.asarray(ds[PET_LON_VARIABLE][:], dtype=np.float64)
	return pet, {d: i for i, d in enumerate(dates)}, lat, lon


def nearest_index(centres, values):
	return np.abs(centres[None, :] - values[:, None]).argmin(axis=1)


def pixel_pet_indices(src, lat, lon):
	"""For each AET pixel, the (lat, lon) index of the PET cell containing its centre."""
	rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
	xs, ys = xy(src.transform, rows.ravel(), cols.ravel())
	converted = warp_transform(src.crs, CRS.from_string(PET_CRS), xs, ys)
	lon_idx = nearest_index(lon, np.asarray(converted[0])).reshape(src.shape)
	lat_idx = nearest_index(lat, np.asarray(converted[1])).reshape(src.shape)
	return lat_idx, lon_idx


def calculate_etf(aet, pet):
	valid = (aet != NODATA) & np.isfinite(aet) & np.isfinite(pet) & (pet > 0)
	etf = np.full(aet.shape, NODATA, dtype=np.float32)
	etf[valid] = aet[valid] / pet[valid]
	return etf

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	files_by_resolution = {
		resolution: aet_files_by_year(resolution)
		for resolution in RESOLUTIONS
	}
	output_dirs = {
		resolution: SCRIPT_DIR / f"Watershed ETf {resolution}"
		for resolution in RESOLUTIONS
	}
	for output_dir in output_dirs.values():
		output_dir.mkdir(parents=True, exist_ok=True)

	counts = {
		resolution: {"written": 0, "skipped": 0}
		for resolution in RESOLUTIONS
	}
	years = sorted({
		year
		for files_by_year in files_by_resolution.values()
		for year in files_by_year
	})
	for year in years:
		pet, date_index, lat, lon = load_pet(year)
		for resolution, files_by_year in files_by_resolution.items():
			files = files_by_year.get(year, [])
			if not files:
				continue
			lat_idx = lon_idx = None
			output_dir = output_dirs[resolution]
			year_dir = output_dir / f"{year}_{resolution}"
			year_dir.mkdir(parents=True, exist_ok=True)
			for date_text, path in tqdm(files, desc=f"{year} {resolution}"):
				if date_text not in date_index:
					counts[resolution]["skipped"] += 1
					continue
				with rasterio.open(path) as src:
					if str(src.crs) != OUTPUT_CRS:
						raise ValueError(f"{path.name} is {src.crs}, expected {OUTPUT_CRS}")
					if lat_idx is None:
						lat_idx, lon_idx = pixel_pet_indices(src, lat, lon)
					aet = src.read(1).astype(np.float64)
					profile = src.profile
				pet_grid = pet[date_index[date_text]][lat_idx, lon_idx]
				etf = calculate_etf(aet, pet_grid)
				profile.update(dtype="float32", nodata=NODATA, compress="deflate")
				output_path = year_dir / f"watershed_ETf_{resolution}_{date_text}.tif"
				with rasterio.open(output_path, "w", **profile) as dst:
					dst.write(etf, 1)
					dst.update_tags(1, units="-", long_name="ET fraction (AET/PET)")
				counts[resolution]["written"] += 1
	for resolution, totals in counts.items():
		print(
			f"Wrote {totals['written']} ETf tifs to {output_dirs[resolution]} "
			f"(skipped {totals['skipped']} without PET)"
		)
