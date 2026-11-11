import re
import numpy as np
import rasterio
from datetime import date
from pathlib import Path
from rasterio.windows import Window
from tqdm import tqdm
from locally_weighted_regression import fill_series, lwr_interpolate
import watershed_data

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ETF_DATASET = "ETf Bhuvan ERA5 Land 750m"
TILE_SIZE = 64
OUTPUT_FILE_TEMPLATE = "watershed_ETf_{resolution}_{date}.tif"
YEAR_DIR_TEMPLATE = "{year}_{resolution}"
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")
NODATA = -99999

POLY_ORDER = 2          # degree of the local polynomial
N_NEIGHBOURS = 7        # nearest valid observations (in time) used per fit
MAX_GAP_DAYS = 30       # gaps with more consecutive missing days are left as nodata
CLIP_MIN = 0.0          # lower bound for interpolated ETf (None to disable)
CLIP_MAX = None         # upper bound for interpolated ETf (None to disable)
WRITE_OBSERVED_DATES = True  # also write dates that already have a raster (pixel-level nodata is filled)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def input_files(config):
	source_dir = SCRIPT_DIR / config.folder
	files = {}
	for path in sorted(source_dir.glob(f"[0-9][0-9][0-9][0-9]_{config.resolution}/{config.pattern}")):
		match = DATE_PATTERN.search(path.name)
		if match:
			text = match.group(1)
			day = date(int(text[:4]), int(text[4:6]), int(text[6:])).toordinal()
			if day in files:
				raise ValueError(f"Duplicate ETf date {text}: {files[day]} and {path}")
			files[day] = path
	if not files:
		raise FileNotFoundError(f"No files matching {config.pattern} in {source_dir}")
	return files


def load_stack(files, days, window, profile, nodata_values):
	arrays = []
	for day in tqdm(days, desc="Reading rasters", unit="raster", leave=False):
		with rasterio.open(files[day]) as src:
			if (src.width, src.height, src.transform, src.crs) != (
				profile["width"], profile["height"], profile["transform"], profile["crs"]
			):
				raise ValueError(f"Raster grid differs from the first input: {files[day]}")
			frame = np.ma.filled(src.read(1, window=window, masked=True).astype(np.float64), np.nan)
			for nodata in nodata_values:
				frame[frame == nodata] = np.nan
			frame[~np.isfinite(frame)] = np.nan
			arrays.append(frame)
	return np.stack(arrays)


def interpolate_stack(stack, days, show_progress=False):
	with tqdm(total=int(np.prod(stack.shape[1:])), desc="Interpolating pixels", unit="pixel",
			  leave=False, disable=not show_progress) as progress:
		all_days, filled = lwr_interpolate(
			days, stack, step=1, poly_order=POLY_ORDER, n_neighbours=N_NEIGHBOURS,
			max_gap=MAX_GAP_DAYS + 1, clip_min=CLIP_MIN, clip_max=CLIP_MAX,
			progress=progress.update if show_progress else None,
		)
	observations = stack.reshape(len(days), -1)
	output = filled.reshape(len(all_days), -1)
	counts = np.isfinite(observations).sum(axis=0)
	sparse_pixels = np.flatnonzero((counts >= 2) & (counts < POLY_ORDER + 1))
	for pixel in tqdm(sparse_pixels, desc="Interpolating sparse pixels", unit="pixel",
					  leave=False, disable=not show_progress or not len(sparse_pixels)):
		output[:, pixel] = fill_series(
			days.astype(np.float64), observations[:, pixel], all_days,
			int(counts[pixel] - 1), N_NEIGHBOURS, MAX_GAP_DAYS + 1, CLIP_MIN, CLIP_MAX,
		)
	for index, day in enumerate(days):
		output_index = int(day - all_days[0])
		valid = np.isfinite(stack[index])
		filled[output_index][valid] = stack[index][valid]
	return filled, all_days


def gapfill_dataset(dataset_name):
	config = watershed_data.DATASETS[dataset_name]
	if not dataset_name.startswith("ETf ") or config.kind != "geotiff":
		raise ValueError(f"Expected a GeoTIFF ETf dataset, got {dataset_name!r}")
	if TILE_SIZE <= 0:
		raise ValueError("TILE_SIZE must be positive")
	files = input_files(config)
	days = np.array(sorted(files))
	with rasterio.open(files[days[0]]) as src:
		profile = dict(src.profile)
	profile.update(dtype="float32", nodata=NODATA, compress="deflate", count=1)
	output_dir = (SCRIPT_DIR / config.folder).with_name(f"{config.folder}_gapfilled")
	observed = set(days.tolist())
	output_paths = {}
	for day in range(int(days[0]), int(days[-1]) + 1):
		if day in observed and not WRITE_OBSERVED_DATES:
			continue
		current = date.fromordinal(day)
		year_dir = output_dir / YEAR_DIR_TEMPLATE.format(year=current.year, resolution=config.resolution)
		year_dir.mkdir(parents=True, exist_ok=True)
		name = OUTPUT_FILE_TEMPLATE.format(resolution=config.resolution, date=current.strftime("%Y%m%d"))
		output_paths[day] = year_dir / name

	windows = [
		Window.from_slices((row, min(row + TILE_SIZE, profile["height"])),
						   (col, min(col + TILE_SIZE, profile["width"])))
		for row in range(0, profile["height"], TILE_SIZE)
		for col in range(0, profile["width"], TILE_SIZE)
	]
	for window_index, window in enumerate(tqdm(windows, desc="Gap-filling tiles", unit="tile")):
		stack = load_stack(files, days, window, profile, config.nodata_values)
		filled, all_days = interpolate_stack(stack, days, show_progress=True)
		for index, day in enumerate(tqdm(all_days, desc="Writing daily rasters", unit="day", leave=False)):
			day = int(day)
			if day not in output_paths:
				continue
			data = np.where(np.isfinite(filled[index]), filled[index], NODATA).astype(np.float32)
			mode = "w" if window_index == 0 else "r+"
			with rasterio.open(output_paths[day], mode, **(profile if mode == "w" else {})) as dst:
				dst.write(data, 1, window=window)
				if mode == "w":
					dst.update_tags(1, units="-", long_name="ET fraction (AET/ref ET), LOESS gap-filled")
	print(f"Wrote {len(output_paths)} ETf tifs to {output_dir}")
	return len(output_paths)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	gapfill_dataset(ETF_DATASET)
