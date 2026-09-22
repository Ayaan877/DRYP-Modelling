import re
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "Kc_FAO56.csv"
OUTPUT_FILE = SCRIPT_DIR / "cropwise_Kc.csv"
MONTH_MAP = {
	"Jan": 1,
	"Feb": 2,
	"Mar": 3,
	"Apr": 4,
	"May": 5,
	"Jun": 6,
	"Jul": 7,
	"Aug": 8,
	"Sep": 9,
	"Oct": 10,
	"Nov": 11,
	"Dec": 12,
}
AREA_COLUMNS = [
	"Kharif_area_(m^2)",
	"Rabi_area_(m^2)",
	"Summer_area_(m^2)",
]
STAGE_HEADERS = [
	("Initial_stage_(days)", "Initial_min_(days)", "Initial_max_(days)"),
	("Dev_stage_(days)", "Dev_min_(days)", "Dev_max_(days)"),
	("Mid_stage_(days)", "Mid_min_(days)", "Mid_max_(days)"),
	("Late_stage_(days)", "Late_min_(days)", "Late_max_(days)"),
]
KC_COLUMNS = ["Kc_ini", "Kc_mid", "Kc_end"]
OUTPUT_COLUMNS = [
	"Class",
	"Season",
	"Crop_type",
	"Sowing_date",
	"Max_area_(m^2)",
	"Growth_period_(days)",
]
OUTPUT_NA_REPRESENTATION = "nan"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def midpoint(min_value, max_value):
	return int(round((min_value + max_value) / 2))


def get_sowing_date(value, month_map):
	if pd.isna(value) or str(value).strip() == "":
		return pd.NA

	start_text = str(value).strip().split("-")[0].strip()
	month_key = re.sub(r"[^A-Za-z]", "", start_text)[:3].capitalize()
	month = month_map.get(month_key)
	if month is None:
		raise ValueError(f"Unsupported sowing month: {value!r}")
	return f"01-{month:02d}"


def clean_Kc_metadata(input_file, stage_headers, area_columns, month_map, kc_columns):
	cleaned_data = pd.read_csv(input_file)
	cleaned_data["Sowing_date"] = cleaned_data["Sowing_period"].apply(
		lambda value: get_sowing_date(value, month_map))
	cleaned_data["Max_area_(m^2)"] = cleaned_data.apply(
		lambda row: row[area_columns].max(), axis=1)

	for stage_name, stage_min, stage_max in stage_headers:
		cleaned_data[stage_name] = cleaned_data.apply(
			lambda row: midpoint(row[stage_min], row[stage_max]), axis=1)

	stage_columns = [stage_name for stage_name, _, _ in stage_headers]
	cleaned_data["Growth_period_(days)"] = cleaned_data[stage_columns].sum(axis=1)
	output_columns = [*OUTPUT_COLUMNS, *stage_columns, *kc_columns]
	return cleaned_data[output_columns]


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def write_kc_metadata():
	cropwise_data = clean_Kc_metadata(
			INPUT_FILE, STAGE_HEADERS, AREA_COLUMNS, MONTH_MAP, KC_COLUMNS)
		
	cropwise_data.to_csv(OUTPUT_FILE, index=False, na_rep=OUTPUT_NA_REPRESENTATION)

	print(f"Wrote {len(cropwise_data)} crops to {OUTPUT_FILE}")

if __name__ == "__main__":
	write_kc_metadata()