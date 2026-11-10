import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import watershed_data
import watershed_operations

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
AET_DATASET = "AET 5500m"
REF_ET_DATASET = "ref ET ERA5"
YEARS = range(2015, 2026)
NODATA = -99999

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def source_name(config, prefix):
	name = config.folder.removeprefix(prefix)
	return name.removesuffix(f" {config.resolution}")


def calculate_etf(aet, ref_et):
	valid = np.isfinite(aet) & np.isfinite(ref_et) & (ref_et > 0)
	etf = np.full(aet.shape, np.nan, dtype=np.float32)
	with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
		np.divide(aet, ref_et, out=etf, where=valid)
	etf[~np.isfinite(etf)] = np.nan
	return etf


def process_year(year, aet_name, ref_et_name):
	aet_config = watershed_data.DATASETS[aet_name]
	ref_et_config = watershed_data.DATASETS[ref_et_name]
	aet = watershed_data.load_data(aet_name, year)
	ref_et = watershed_data.load_data(ref_et_name, year)
	if aet.units != ref_et.units:
		raise ValueError(f"ET units must match: {aet.units} and {ref_et.units}")
	dates = sorted(set(aet.dates).intersection(ref_et.dates))
	if not dates:
		print(f"{year}: no shared observation dates")
		return 0

	finest_config = aet_config if aet.resolution_m <= ref_et.resolution_m else ref_et_config
	resolution = finest_config.resolution
	transform, width, height = watershed_operations.target_grid([aet, ref_et])
	output_dir = SCRIPT_DIR / (
		f"Watershed ETf {source_name(aet_config, 'Watershed AET ')}-"
		f"{source_name(ref_et_config, 'Watershed ref ET ')} {resolution}"
	) / f"{year}_{resolution}"
	output_dir.mkdir(parents=True, exist_ok=True)
	profile = {
		"driver": "GTiff", "width": width, "height": height, "count": 1,
		"dtype": "float32", "crs": watershed_operations.PROJECT_CRS,
		"transform": transform, "nodata": NODATA, "compress": "deflate",
	}
	for current in tqdm(dates, desc=f"{year} {resolution}"):
		aet_frame = watershed_operations.align_series(aet, [current], transform, width, height)[0]
		ref_et_frame = watershed_operations.align_series(ref_et, [current], transform, width, height)[0]
		etf = calculate_etf(aet_frame, ref_et_frame)
		data = np.where(np.isfinite(etf), etf, NODATA).astype(np.float32)
		output_path = output_dir / f"watershed_ETf_{resolution}_{current:%Y%m%d}.tif"
		with rasterio.open(output_path, "w", **profile) as dst:
			dst.write(data, 1)
			dst.update_tags(1, units="-", long_name="ET fraction (AET/ref ET)")
	print(f"{year}: wrote {len(dates)} ETf rasters to {output_dir}")
	return len(dates)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	for year in YEARS:
		try:
			process_year(year, AET_DATASET, REF_ET_DATASET)
		except FileNotFoundError as error:
			print(f"{year}: skipped ({error})")
