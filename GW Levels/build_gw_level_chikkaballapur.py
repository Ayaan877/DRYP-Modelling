import re
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ABY_INPUT_PATH = SCRIPT_DIR / "ABY-KWRIS_Combined.csv"
CGWB_INPUT_PATH = SCRIPT_DIR / "CGWB_Combined.csv"
OUTPUT_PATH = SCRIPT_DIR / "ABY-KWRIS-CGWB_Combined.csv"

OUTPUT_COLUMNS = [
	"source",
	"latitude",
	"longitude",
	"well ID",
	"village",
	"block",
	"well type",
	"status",
	"well depth",
	"well depth units",
	"water level",
	"water level units",
	"collection date",
	"collection time",
]

ABY_DEPTH_UNIT = "mbmp"
ABY_LEVEL_UNIT = "mbgl"
MISSING_VALUES = {"", "-"}
STATUS_WET = "wet"
STATUS_DRY = "dry"
DRY_TOKEN = "dry"
SUFFIX_PATTERN = re.compile(r"(_?\d+)$")

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def read_csv(path: Path):
	"""Read a combined CSV with every field as text."""
	return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_name(value: str):
	"""Capitalize a name in lowercase, keeping any trailing numeric suffix."""
	text = value.strip()
	if text in MISSING_VALUES:
		return ""
	match = SUFFIX_PATTERN.search(text)
	base, suffix = (text[: match.start()], match.group(1)) if match else (text, "")
	base = base.strip().lower()
	return base[:1].upper() + base[1:] + suffix


def source_from_path(path: Path):
	"""Derive the row source from the input file name, e.g. ABY-KWRIS_Combined.csv."""
	return path.stem.removesuffix("_Combined")


def water_status(level: pd.Series, raw_level: pd.Series):
	"""Label rows wet when a level is read and dry when the source says dry."""
	status = pd.Series("", index=level.index)
	status = status.mask(level.notna(), STATUS_WET)
	return status.mask(raw_level.str.strip().str.lower() == DRY_TOKEN, STATUS_DRY)


def load_aby(path: Path):
	"""Map the ABY combined sheet onto the output columns."""
	raw = read_csv(path)
	depth = pd.to_numeric(raw["Depth of Well(mbmp)"], errors="coerce")
	level = pd.to_numeric(raw["Ground Water Level(mbgl)"], errors="coerce")
	dates = pd.to_datetime(raw["Ground Water Level Date"], errors="coerce")
	status = water_status(level, raw["Ground Water Level(mbgl)"])
	return pd.DataFrame(
		{
			"source": pd.Series(source_from_path(path), index=raw.index),
			"status": status,
			"latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
			"longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
			"well ID": raw["Well ID"],
			"village": raw["Village Name"].map(normalize_name),
			"block": raw["Block"].map(normalize_name),
			"well type": raw["Type of Well"].map(normalize_name),
			"well depth": depth,
			"well depth units": pd.Series(ABY_DEPTH_UNIT, index=raw.index).where(depth.notna(), ""),
			"water level": level,
			"water level units": pd.Series(ABY_LEVEL_UNIT, index=raw.index).where(level.notna(), ""),
			"collection date": dates.dt.strftime("%Y-%m-%d"),
			"collection time": pd.Series("", index=raw.index),
		},
		columns=OUTPUT_COLUMNS,
	)


def load_cgwb(path: Path):
	"""Map the CGWB combined sheet onto the output columns."""
	raw = read_csv(path)
	times = pd.to_datetime(raw["dataTime"], errors="coerce")
	level = pd.to_numeric(raw["dataValue"], errors="coerce")
	status = water_status(level, raw["dataValue"])
	return pd.DataFrame(
		{
			"source": pd.Series(source_from_path(path), index=raw.index),
			"status": status,
			"latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
			"longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
			"well ID": raw["stationCode"],
			"village": raw["stationName"].map(normalize_name),
			"block": raw["tehsil"].map(normalize_name),
			"well type": raw["wellType"].map(normalize_name),
			"well depth": pd.to_numeric(raw["wellDepth"], errors="coerce"),
			"well depth units": pd.Series("", index=raw.index),
			"water level": pd.to_numeric(raw["dataValue"], errors="coerce"),
			"water level units": pd.Series("", index=raw.index),
			"collection date": times.dt.strftime("%Y-%m-%d"),
			"collection time": times.dt.strftime("%H:%M:%S"),
		},
		columns=OUTPUT_COLUMNS,
	)


def build_gw_level(aby_path: Path, cgwb_path: Path):
	"""Stack the ABY and CGWB rows and check the row count against the inputs."""
	print(f"Entries in ABY-KWRIS: {len(read_csv(aby_path))} | Entries in CGWB: {len(read_csv(cgwb_path))}")
	expected_rows = len(read_csv(aby_path)) + len(read_csv(cgwb_path))
	frame = pd.concat([load_aby(aby_path), load_cgwb(cgwb_path)], ignore_index=True)
	print(f"Entries in cleaned data: {len(frame)}")
	if len(frame) != expected_rows:
		raise ValueError(f"Expected {expected_rows} rows from the inputs, got {len(frame)}")
	else:
		print("Row count matches expected value.")
	return frame


def write_gw_level(frame: pd.DataFrame, path: Path):
	"""Write the combined table without the index."""
	frame.to_csv(path, index=False, encoding="utf-8")


def main():
	frame = build_gw_level(ABY_INPUT_PATH, CGWB_INPUT_PATH)
	write_gw_level(frame, OUTPUT_PATH)
	print(f"Wrote {len(frame)} rows to {OUTPUT_PATH.name}")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	main()
	
