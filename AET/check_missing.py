import csv
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATASET = "SSEBop"  # "Bhuvan" or "SSEBop"
RESOLUTION = "750m"
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 12, 31)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_CONFIGS = {
	"Bhuvan": {
		"input_dir": f"Bhuvan AET {RESOLUTION} - Extracted",
		"output_prefix": f"Bhuvan{RESOLUTION}",
		"period_type": "daily",
		"period_label": "days",
	},
	"SSEBop": {
		"input_dir": "SSEBop Dekadal AET - Extracted",
		"output_prefix": "SSEBopDekadal",
		"period_type": "dekadal",
		"period_label": "dekads",
	},
}
if DATASET not in DATASET_CONFIGS:
	raise ValueError(f"DATASET must be one of {', '.join(DATASET_CONFIGS)}")
DATASET_CONFIG = DATASET_CONFIGS[DATASET]

INPUT_DIR = SCRIPT_DIR / DATASET_CONFIG["input_dir"]

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_PREFIX = DATASET_CONFIG["output_prefix"]
COUNT_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_missing_count.csv"
MATRIX_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_missing_matrix.csv"
ZERO_BYTE_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_zero_byte.csv"

DATE_FORMAT = re.compile(r"(?<!\d)(?:19|20)\d{6}(?!\d)")
SSEBOP_DATE_FORMAT = re.compile(
	r"\bd(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<dekad>[1-3])(?!\d)"
)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def get_date(token):
	try:
		return dt.datetime.strptime(token, "%Y%m%d").date()
	except ValueError:
		return None


def iter_date_range(start, end):
	days = (end - start).days
	for offset in range(days + 1):
		yield start + dt.timedelta(days=offset)


def iter_dekadal_range(start, end):
	current = dt.date(start.year, start.month, 1)
	while current <= end:
		for day in (1, 11, 21):
			period = dt.date(current.year, current.month, day)
			if start <= period <= end:
				yield period
		if current.month == 12:
			current = dt.date(current.year + 1, 1, 1)
		else:
			current = dt.date(current.year, current.month + 1, 1)


def extract_member_date(member_name, period_type):
	filename = Path(member_name).name
	if period_type == "daily":
		tokens = DATE_FORMAT.findall(filename)
		if not tokens:
			return None
		return get_date(tokens[-1])

	match = SSEBOP_DATE_FORMAT.search(filename)
	if not match:
		return None
	try:
		day = (1, 11, 21)[int(match.group("dekad")) - 1]
		return dt.date(int(match.group("year")), int(match.group("month")), day)
	except ValueError:
		return None


def collect_extracted_data(input_dir, period_type):
	if period_type == "daily":
		expected_dates = set(iter_date_range(START_DATE, END_DATE))
	else:
		expected_dates = set(iter_dekadal_range(START_DATE, END_DATE))
	present_dates = set()
	zero_byte_dates = set()
	tiff_count = 0

	for tif_path in input_dir.rglob("*"):
		if not tif_path.is_file() or tif_path.suffix.lower() != ".tif":
			continue
		tiff_count += 1
		file_date = extract_member_date(tif_path.name, period_type)
		if not file_date or file_date not in expected_dates:
			continue
		if tif_path.stat().st_size == 0:
			zero_byte_dates.add(file_date)
		else:
			present_dates.add(file_date)

	return expected_dates, present_dates, zero_byte_dates, tiff_count


def count_missing_periods(missing_dates, expected_dates, output_path, period_label):
	counts = defaultdict(int)
	for day in missing_dates:
		counts[day.year] += 1

	years = sorted({day.year for day in expected_dates})
	with output_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["year", f"missing_{period_label}"])
		for year in years:
			writer.writerow([year, counts[year]])


def write_missing_matrix(expected_dates, present_dates, output_path, period_type):
	years_months = sorted({(day.year, day.month) for day in expected_dates})
	with output_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		if period_type == "dekadal":
			writer.writerow(["year", "month", "dekad_1", "dekad_2", "dekad_3"])
			for year, month in years_months:
				row = [year, month]
				for day in (1, 11, 21):
					current = dt.date(year, month, day)
					row.append(current in present_dates if current in expected_dates else "Null")
				writer.writerow(row)
			return

		writer.writerow(["year", "month", *range(1, 32)])
		for year, month in years_months:
			row = [year, month]
			for day in range(1, 32):
				try:
					current = dt.date(year, month, day)
				except ValueError:
					row.append("Null")
					continue

				if current in expected_dates:
					row.append(current in present_dates)
				else:
					row.append("Null")
			writer.writerow(row)


def write_date_list(path, dates, period_type):
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["dekad_start_date" if period_type == "dekadal" else "date"])
		for day in sorted(dates):
			writer.writerow([day.isoformat()])


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def run_missing_date_checker():
	if DATASET == "Bhuvan" and RESOLUTION not in {"5500m", "750m"}:
		raise ValueError('RESOLUTION must be set to "5500m" or "750m"')
	if not INPUT_DIR.is_dir():
		raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

	expected_dates, present_dates, zero_byte_dates, tiff_count = (
		collect_extracted_data(INPUT_DIR, DATASET_CONFIG["period_type"])
	)
	if not tiff_count:
		raise RuntimeError(f"No TIFF files found in {INPUT_DIR}")

	missing_dates = (expected_dates - present_dates) | zero_byte_dates

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	count_missing_periods(
		missing_dates, expected_dates, COUNT_OUTPUT, DATASET_CONFIG["period_label"]
	)
	write_missing_matrix(
		expected_dates, present_dates, MATRIX_OUTPUT, DATASET_CONFIG["period_type"]
	)
	write_date_list(ZERO_BYTE_OUTPUT, zero_byte_dates, DATASET_CONFIG["period_type"])

	print(f"Dataset: {DATASET}")
	if DATASET == "Bhuvan":
		print(f"Resolution: {RESOLUTION}")
	print(f"Scanned extracted folder: {INPUT_DIR}")
	print(f"TIFF files found: {tiff_count}")
	print(f"Expected {DATASET_CONFIG['period_label']}: {len(expected_dates)}")
	print(f"Present TIFF {DATASET_CONFIG['period_label']}: {len(present_dates & expected_dates)}")
	print(f"Zero-byte TIFF {DATASET_CONFIG['period_label']}: {len(zero_byte_dates & expected_dates)}")
	print(f"Missing {DATASET_CONFIG['period_label']}: {len(missing_dates)}")
	print(f"Wrote: {COUNT_OUTPUT}")
	print(f"Wrote: {MATRIX_OUTPUT}")
	print(f"Wrote: {ZERO_BYTE_OUTPUT}")
	print("Matrix CSV meaning: True=present data, False=absent data, Null=non-applicable period.")


if __name__ == "__main__":
	run_missing_date_checker()
