import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ALLOCATION_FILE = SCRIPT_DIR / "HN_valley_allocation.csv"
OUTPUT_FILE = SCRIPT_DIR / "HN_valley_hourly_inflow.csv"

TIME_ORIGIN = "2015-01-01 00:00:00"
TIME_END = "2025-12-31 23:00:00"
EXCLUDED_TANK_NUMBER = 1
DRY_MONTHS = (6, 7, 8, 9)
ALLOCATION_FRAC = 0.5
DISCHARGE_START = "2022-01-01 00:00:00"
GRID_CELL_AREA_M2 = 250 * 250

TANK_NUMBER_COLUMN = "Tank No."
ALLOCATION_COLUMN = "Water Allocated (MLD)"
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
FLUX_COLUMN = "HN_flux"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_eligible_allocation(allocation_file):
	data = pd.read_csv(allocation_file)
	required_columns = {TANK_NUMBER_COLUMN, ALLOCATION_COLUMN}
	if not required_columns.issubset(data.columns):
		raise ValueError(
			"Allocation file must contain 'Tank No.' and 'Water Allocated (MLD)' columns")

	eligible_tanks = data[TANK_NUMBER_COLUMN] != EXCLUDED_TANK_NUMBER
	return data.loc[eligible_tanks, ALLOCATION_COLUMN].sum()


def make_hourly_inflow(allocation_file):
	allocation_mld = read_eligible_allocation(allocation_file)
	hours = pd.date_range(TIME_ORIGIN, TIME_END, freq="h")
	inflow = pd.DataFrame({
		DATE_COLUMN: hours.strftime("%d/%m/%Y"),
		TIME_COLUMN: hours.strftime("%H:%M:%S"),
	})

	hourly_allocation_mm = (
		allocation_mld
		* ALLOCATION_FRAC
		* 1000  # ML/day to m3/day
		/ 24
		/ GRID_CELL_AREA_M2
		* 1000  # metres to millimetres
	)
	inflow[FLUX_COLUMN] = hourly_allocation_mm
	inflow.loc[
		(hours < pd.Timestamp(DISCHARGE_START))
		| hours.month.isin(DRY_MONTHS),
		FLUX_COLUMN,
	] = 0

	return inflow


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def write_hourly_inflow():
	hn_inflow = make_hourly_inflow(ALLOCATION_FILE)
	hn_inflow.to_csv(OUTPUT_FILE, index=False, float_format="%.7f")

	print(f"Saved hourly HN Valley inflow records for {len(hn_inflow)} hours to {OUTPUT_FILE.name}")
	print(f"Period: {TIME_ORIGIN} to {TIME_END}")
	print(f"Nonzero flux value = {hn_inflow[FLUX_COLUMN].max()} mm/hr/cell")\
	
if __name__ == "__main__":
	write_hourly_inflow()
