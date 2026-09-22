import numpy as np
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

VERSION = "v2.0"
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / f"Rooting_Depth_{VERSION}_10m.asc"
OUTPUT_FILE = SCRIPT_DIR / f"Rooting_Depth_{VERSION}_250m.asc"

TARGET_CRS = "EPSG:32643"
TARGET_NCOLS = 156
TARGET_NROWS = 105
TARGET_XLLCORNER = 798778.801047321060
TARGET_YLLCORNER = 1484563.296254347777
TARGET_CELLSIZE = 250.0
NODATA = -99999

TARGET_CRS_OBJECT = CRS.from_string(TARGET_CRS)
TARGET_TRANSFORM = from_origin(
	TARGET_XLLCORNER,
	TARGET_YLLCORNER + TARGET_NROWS * TARGET_CELLSIZE,
	TARGET_CELLSIZE,
	TARGET_CELLSIZE)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def regrid(source, source_transform, source_crs, source_nodata):
	target = np.full((TARGET_NROWS, TARGET_NCOLS), NODATA, dtype=np.float32)
	reproject(
		source=np.asarray(source, dtype=np.float32),
		destination=target,
		src_transform=source_transform,
		src_crs=source_crs,
		src_nodata=source_nodata,
		dst_transform=TARGET_TRANSFORM,
		dst_crs=TARGET_CRS_OBJECT,
		dst_nodata=NODATA,
		resampling=Resampling.average)
	return target


def write_ascii_grid(output_file, grid):
	with output_file.open("w", encoding="ascii", newline="\n") as output:
		output.write(f"ncols        {TARGET_NCOLS}\n")
		output.write(f"nrows        {TARGET_NROWS}\n")
		output.write(f"xllcorner    {TARGET_XLLCORNER:.12f}\n")
		output.write(f"yllcorner    {TARGET_YLLCORNER:.12f}\n")
		output.write(f"cellsize     {TARGET_CELLSIZE:.12f}\n")
		output.write(f"NODATA_value {NODATA}\n")
		for row in grid:
			output.write(" ".join(f"{value:.6g}" for value in row) + "\n")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def downscale_rooting_depth(input_file, output_file):
	with rasterio.open(input_file) as source:
		depth = source.read(1, masked=True)
		source_nodata = source.nodata if source.nodata is not None else NODATA
		source_crs = source.crs or TARGET_CRS_OBJECT
		target = regrid(
			depth.filled(source_nodata),
			source.transform,
			source_crs,
			source_nodata,
		)
	write_ascii_grid(output_file, target)
	print(f"Wrote {output_file}: {TARGET_NROWS} x {TARGET_NCOLS}")


if __name__ == "__main__":
	downscale_rooting_depth(INPUT_FILE, OUTPUT_FILE)