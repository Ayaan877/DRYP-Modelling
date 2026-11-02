import fiona
import rasterio
import re
import zipfile
from pathlib import Path
from rasterio.errors import RasterioIOError
from rasterio.mask import mask
from rasterio.warp import transform_geom

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESOLUTION_LABEL = "750m"  # "5500m" or "750m"
INPUT_ROOT = SCRIPT_DIR / f"Bhuvan AET {RESOLUTION_LABEL} - Extracted"
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
OUTPUT_DIR = SCRIPT_DIR / f"Watershed AET {RESOLUTION_LABEL}"
RAW_DIR = SCRIPT_DIR / "Bhuvan AET Raw" / f"AET {RESOLUTION_LABEL}"
SOURCE_CRS = "EPSG:4326"
WATERSHED_CRS = "EPSG:32643"
DATE_PATTERN = re.compile(r"(\d{8})")

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_watershed():
	with fiona.open(WATERSHED_FILE) as src:
		return [feature["geometry"] for feature in src]


def reproject_watershed(watershed_shape):
	return [transform_geom(WATERSHED_CRS, SOURCE_CRS, geom) 
	        for geom in watershed_shape]


def mask_raster(raster_file, reprojected_watershed):
	with rasterio.open(raster_file) as src:
		clipped, transform = mask(
			src,
			shapes=reprojected_watershed,
			crop=True,
			all_touched=True,
		)
		meta = src.meta.copy()
		meta.update(
			{
				"height": clipped.shape[1],
				"width": clipped.shape[2],
				"transform": transform,
			}
		)
	return clipped, meta


def save_masked_raster(source_file, clipped_data, output_meta):
	date_match = DATE_PATTERN.search(source_file.stem)
	if not date_match:
		raise ValueError(f"No date found in file name: {source_file.name}")

	file_date = date_match.group(1)
	year = file_date[:4]
	target_folder = OUTPUT_DIR / f"{year}_{RESOLUTION_LABEL}"
	target_file = target_folder / f"watershed_bhuvan_AET_{RESOLUTION_LABEL}_{file_date}.tif"
	target_file.parent.mkdir(parents=True, exist_ok=True)
	with rasterio.open(target_file, "w", **output_meta) as dst:
		dst.write(clipped_data)
	return target_file


def raw_zip_dates():
	"""Dates of every .tif listed inside the raw zip files (central directory only, no extraction)."""
	dates = []
	for zip_path in sorted(RAW_DIR.glob("*.zip")):
		with zipfile.ZipFile(zip_path) as zf:
			for name in zf.namelist():
				if name.lower().endswith(".tif"):
					match = DATE_PATTERN.search(Path(name).stem)
					dates.append(match.group(1) if match else f"UNKNOWN:{name}")
	return dates


def validate_outputs():
	raw_dates = raw_zip_dates()
	out_dates = [
		match.group(1)
		for p in OUTPUT_DIR.rglob("*.tif")
		if (match := DATE_PATTERN.search(p.stem))
	]
	raw_set, out_set = set(raw_dates), set(out_dates)
	missing = sorted(raw_set - out_set)
	extra = sorted(out_set - raw_set)

	print("\nValidation (raw zips vs. output)")
	print(f"  Raw tif files in zips : {len(raw_dates)} ({len(raw_set)} unique dates)")
	print(f"  Output tif files      : {len(out_dates)} ({len(out_set)} unique dates)")
	print(f"  Missing from output   : {len(missing)} {missing[:20]}")
	print(f"  Not in raw zips       : {len(extra)} {extra[:20]}")
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

	tif_files = [
		path for path in INPUT_ROOT.rglob("*.tif")
		if OUTPUT_DIR not in path.parents
	]
	tif_files.sort()

	print(f"Found {len(tif_files)} tif files")
	skipped_zero_byte = 0
	corrupt_files = []

	for i, tif_file in enumerate(tif_files, start=1):
		if tif_file.stat().st_size == 0:
			skipped_zero_byte += 1
			print(f"[{i}/{len(tif_files)}] Skipped zero-byte file: {tif_file}")
			continue

		try:
			clipped_data, output_meta = mask_raster(tif_file, reprojected_watershed)
		except RasterioIOError as err:
			corrupt_files.append(tif_file)
			print(f"[{i}/{len(tif_files)}] Skipped unreadable file: {tif_file} ({err})")
			continue
		saved_file = save_masked_raster(tif_file, clipped_data, output_meta)
		print(f"[{i}/{len(tif_files)}] Saved {saved_file}")

	print(f"Skipped zero-byte files: {skipped_zero_byte}")
	print(f"Skipped unreadable files: {len(corrupt_files)}")
	for path in corrupt_files:
		print(f"  {path}")
	print(f"Done. Watershed rasters saved in: {OUTPUT_DIR}")
	validate_outputs()