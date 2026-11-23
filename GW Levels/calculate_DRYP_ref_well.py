from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEM_PATH = SCRIPT_DIR / "DRYP Data" / "DRYP_ref_well_DEM.csv"
WTE_PATH = SCRIPT_DIR / "DRYP Data" / "DRYP_ref_well_wte.csv"
DATA_OUT_PATH = SCRIPT_DIR / "DRYP_ref_well_data.csv"
SUMMARY_OUT_PATH = SCRIPT_DIR / "DRYP_ref_well_summary.csv"

WTE_DATE_FORMAT = "%d-%m-%Y %H:%M"
DEM_COLUMN = "DEM elevation"
SOURCE_NAME = "DRYP"
STATUS_WET = "wet"
WATER_LEVEL_UNITS = "m"

DATA_COLUMNS = [
	"source",
	"latitude",
	"longitude",
	"well ID",
	"status",
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


def load_wells(path: Path):
	"""Read the reference well metadata and ground elevation from the DEM table."""
	frame = read_csv(path)
	return frame[["latitude", "longitude", "well ID", DEM_COLUMN]]


def load_wte(path: Path):
	"""Read the hourly water table elevation, indexed by timestamp."""
	frame = read_csv(path)
	frame.index = pd.to_datetime(frame.pop("Date"), format=WTE_DATE_FORMAT)
	return frame


def build_data(wells: pd.DataFrame, wte: pd.DataFrame):
	"""Compute water level as DEM elevation minus water table elevation for each well and hour."""
	frames = []
	for _, well in wells.iterrows():
		well_id = well["well ID"]
		level =  pd.to_numeric(wte[well_id], errors="coerce") - float(well[DEM_COLUMN])
		frame = pd.DataFrame({"time": wte.index, "water level": level.to_numpy()}).dropna(subset=["water level"])
		frame["source"] = SOURCE_NAME
		frame["latitude"] = well["latitude"]
		frame["longitude"] = well["longitude"]
		frame["well ID"] = well_id
		frame["status"] = STATUS_WET
		frame["water level units"] = WATER_LEVEL_UNITS
		frame["collection date"] = frame["time"].dt.strftime("%Y-%m-%d")
		frame["collection time"] = frame["time"].dt.strftime("%H:%M:%S")
		frames.append(frame[DATA_COLUMNS])
	return pd.concat(frames, ignore_index=True)


def build_summary(data: pd.DataFrame, wells: pd.DataFrame):
	"""Summarise one row per well with its reading count and monitoring period."""
	dates = pd.to_datetime(data["collection date"], errors="coerce")
	grouped = data.assign(_date=dates).groupby("well ID", sort=False)
	summary = grouped.agg(
		source=("source", "first"),
		latitude=("latitude", "first"),
		longitude=("longitude", "first"),
		reading_count=("water level", "size"),
		monitoring_start=("_date", "min"),
		monitoring_stop=("_date", "max"),
	).reset_index()
	summary = summary.rename(columns={
		"reading_count": "reading count",
		"monitoring_start": "monitoring start",
		"monitoring_stop": "monitoring stop",
	})
	for column in ["monitoring start", "monitoring stop"]:
		summary[column] = summary[column].dt.strftime("%Y-%m-%d")
	summary = summary.set_index("well ID").reindex(wells["well ID"]).reset_index()
	return summary[SUMMARY_COLUMNS]


def main():
	wells = load_wells(DEM_PATH)
	wte = load_wte(WTE_PATH)
	data = build_data(wells, wte)
	summary = build_summary(data, wells)
	data.to_csv(DATA_OUT_PATH, index=False, encoding="utf-8")
	summary.to_csv(SUMMARY_OUT_PATH, index=False, encoding="utf-8")
	print(f"Wrote {len(data)} rows to {DATA_OUT_PATH.name}")
	print(f"Wrote {len(summary)} wells to {SUMMARY_OUT_PATH.name}")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	main()
