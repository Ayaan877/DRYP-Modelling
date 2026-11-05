import re
import numpy as np
import rasterio
from datetime import date
from pathlib import Path
from tqdm import tqdm
from locally_weighted_regression import lwr_interpolate

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESOLUTION = "750m"
SOURCE_DIR = SCRIPT_DIR / f"Watershed ETf {RESOLUTION}"
FILE_PATTERN = f"watershed_ETf_{RESOLUTION}_*.tif"
OUTPUT_DIR = SOURCE_DIR.with_name(f"{SOURCE_DIR.name} - gapfilled")
OUTPUT_FILE_TEMPLATE = "watershed_ETf_{resolution}_{date}.tif"
YEAR_DIR_TEMPLATE = "{year}_{resolution}"
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")
NODATA = -99999

POLY_ORDER = 2          # degree of the local polynomial
N_NEIGHBOURS = 7        # nearest valid observations (in time) used per fit
MAX_GAP_DAYS = 365       # gaps with more consecutive missing days are left as nodata
CLIP_MIN = 0.0          # lower bound for interpolated ETf (None to disable)
CLIP_MAX = None         # upper bound for interpolated ETf (None to disable)
WRITE_OBSERVED_DATES = True  # also write dates that already have a raster (pixel-level nodata is filled)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_stack():
	"""Observed ETf stack (n_obs, H, W), their day numbers, and a reference profile."""
	files = {}
	for path in sorted(SOURCE_DIR.rglob(FILE_PATTERN)):
		match = DATE_PATTERN.search(path.name)
		if match:
			text = match.group(1)
			files[date(int(text[:4]), int(text[4:6]), int(text[6:])).toordinal()] = path
	if not files:
		raise FileNotFoundError(f"No files matching {FILE_PATTERN} in {SOURCE_DIR}")
	days = np.array(sorted(files))
	arrays = []
	with rasterio.open(files[days[0]]) as src:
		profile = dict(src.profile)
	for day in tqdm(days, desc="Reading"):
		with rasterio.open(files[day]) as src:
			arrays.append(src.read(1).astype(np.float64))
	stack = np.stack(arrays)
	stack[(stack == NODATA) | ~np.isfinite(stack)] = np.nan
	return stack, days, profile


def interpolate_stack(stack, days):
	return lwr_interpolate(days, stack, step=1, poly_order=POLY_ORDER, n_neighbours=N_NEIGHBOURS,
						   max_gap=MAX_GAP_DAYS + 1, clip_min=CLIP_MIN, clip_max=CLIP_MAX)[::-1]

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	stack, days, profile = load_stack()
	filled, all_days = interpolate_stack(stack, days)
	observed = set(days.tolist())
	profile.update(dtype="float32", nodata=NODATA, compress="deflate")
	written = 0
	for index, day in enumerate(tqdm(all_days, desc="Writing")):
		if day in observed and not WRITE_OBSERVED_DATES:
			continue
		current = date.fromordinal(int(day))
		year_dir = OUTPUT_DIR / YEAR_DIR_TEMPLATE.format(year=current.year, resolution=RESOLUTION)
		year_dir.mkdir(parents=True, exist_ok=True)
		data = np.where(np.isnan(filled[index]), NODATA, filled[index]).astype(np.float32)
		name = OUTPUT_FILE_TEMPLATE.format(resolution=RESOLUTION, date=current.strftime("%Y%m%d"))
		with rasterio.open(year_dir / name, "w", **profile) as dst:
			dst.write(data, 1)
			dst.update_tags(1, units="-", long_name="ET fraction (AET/PET), LOESS gap-filled")
		written += 1
	print(f"Wrote {written} ETf tifs to {OUTPUT_DIR}")
