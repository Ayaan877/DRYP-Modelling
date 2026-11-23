from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINED_PATH = SCRIPT_DIR / "ABY-KWRIS-CGWB_Combined.csv"
DATA_OUT_PATH = SCRIPT_DIR / "watershed_ref_well_data.csv"
SUMMARY_OUT_PATH = SCRIPT_DIR / "watershed_ref_well_summary.csv"

CGWB_WELL_IDS = ["CGWBNG0170", "CGWBNG0169"]
KWRIS_WELL_IDS = ["131013", "131002", "131004"]
WELL_IDS = CGWB_WELL_IDS + KWRIS_WELL_IDS

DATA_COLUMNS = [
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

SUMMARY_COLUMNS = [
	"source",
	"latitude",
	"longitude",
	"well ID",
	"village",
	"block",
	"well type",
	"well depth",
	"reading count",
	"monitoring start",
	"monitoring stop",
]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def read_csv(path: Path):
	"""Read a CSV with every field as text."""
	return pd.read_csv(path, dtype=str, keep_default_na=False)


def select_wells(path: Path, well_ids, source: str):
	"""Keep rows from one source for the requested wells that have a numeric water level."""
	frame = read_csv(path)
	frame = frame[(frame["source"] == source) & frame["well ID"].isin(well_ids)]
	level = pd.to_numeric(frame["water level"], errors="coerce")
	return frame[level.notna()]


def drop_positive_levels(frame: pd.DataFrame):
	"""Keep only rows with a zero or negative water level."""
	return frame[pd.to_numeric(frame["water level"]) <= 0]


def build_data(frames):
	"""Stack the selected rows in the order of the input files."""
	return pd.concat(frames, ignore_index=True)[DATA_COLUMNS]


def build_summary(data: pd.DataFrame):
	"""Summarise one row per well with its reading count and monitoring period."""
	dates = pd.to_datetime(data["collection date"], errors="coerce")
	grouped = data.assign(_date=dates).groupby("well ID", sort=False)
	summary = grouped.agg(
		source=("source", "first"),
		latitude=("latitude", "first"),
		longitude=("longitude", "first"),
		village=("village", "first"),
		block=("block", "first"),
		well_type=("well type", "first"),
		well_depth=("well depth", "first"),
		reading_count=("water level", "size"),
		monitoring_start=("_date", "min"),
		monitoring_stop=("_date", "max"),
	).reset_index()
	summary = summary.rename(columns={
		"well_type": "well type",
		"well_depth": "well depth",
		"reading_count": "reading count",
		"monitoring_start": "monitoring start",
		"monitoring_stop": "monitoring stop",
	})
	for column in ["monitoring start", "monitoring stop"]:
		summary[column] = summary[column].dt.strftime("%Y-%m-%d")
	summary = summary.set_index("well ID").reindex(WELL_IDS).reset_index()
	return summary[SUMMARY_COLUMNS]


def main():
	cgwb_frame = drop_positive_levels(select_wells(COMBINED_PATH, CGWB_WELL_IDS, "CGWB"))
	kwris_frame = select_wells(COMBINED_PATH, KWRIS_WELL_IDS, "KWRIS")
	data = build_data([cgwb_frame, kwris_frame])
	summary = build_summary(data)
	data.to_csv(DATA_OUT_PATH, index=False, encoding="utf-8")
	summary.to_csv(SUMMARY_OUT_PATH, index=False, encoding="utf-8")
	print(f"Wrote {len(data)} rows to {DATA_OUT_PATH.name}")
	print(f"Wrote {len(summary)} wells to {SUMMARY_OUT_PATH.name}")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	main()
