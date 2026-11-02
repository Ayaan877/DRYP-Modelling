import csv
import datetime as dt
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESOLUTION = "5500m"
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 12, 31)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Bhuvan AET Raw" / f"AET {RESOLUTION}"

OUTPUT_DIR = SCRIPT_DIR
COUNT_OUTPUT = OUTPUT_DIR / f"Bhuvan{RESOLUTION}_missing_count.csv"
MATRIX_OUTPUT = OUTPUT_DIR / f"Bhuvan{RESOLUTION}_missing_matrix.csv"
ZERO_BYTE_OUTPUT = OUTPUT_DIR / f"Bhuvan{RESOLUTION}_zero_byte.csv"

ZIP_RANGE_FORMAT = re.compile(r"_(\d{8})_(\d{8})\.zip$")
DATE_FORMAT = re.compile(r"(?<!\d)(?:19|20)\d{6}(?!\d)")

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


def find_zip_ranges(input_dir):
	ranges = []
	for zip_path in sorted(input_dir.glob("*.zip")):
		match = ZIP_RANGE_FORMAT.search(zip_path.name)
		if not match:
			continue

		start = get_date(match.group(1))
		end = get_date(match.group(2))
		if not start or not end:
			continue
		if start > end:
			start, end = end, start
		ranges.append((zip_path, start, end))
	return ranges


def extract_member_date(member_name):
	tokens = DATE_FORMAT.findall(Path(member_name).name)
	if not tokens:
		return None
	return get_date(tokens[-1])


def collect_zip_data(zip_ranges):
	expected_dates = set(iter_date_range(START_DATE, END_DATE))
	present_dates = set()
	zero_byte_dates = set()

	for zip_path, start, end in zip_ranges:
		with zipfile.ZipFile(zip_path) as archive:
			for member in archive.infolist():
				if not member.filename.lower().endswith(".tif"):
					continue
				file_date = extract_member_date(member.filename)
				if not file_date or file_date not in expected_dates:
					continue
				if member.file_size == 0:
					zero_byte_dates.add(file_date)
				else:
					present_dates.add(file_date)

	return expected_dates, present_dates, zero_byte_dates


def count_missing_days(missing_dates, expected_dates):
	counts = defaultdict(int)
	for day in missing_dates:
		counts[day.year] += 1

	years = sorted({day.year for day in expected_dates})
	with COUNT_OUTPUT.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["year", "missing_days"])
		for year in years:
			writer.writerow([year, counts[year]])


def write_missing_matrix(expected_dates, present_dates):
	years_months = sorted({(day.year, day.month) for day in expected_dates})
	with MATRIX_OUTPUT.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
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


def write_date_list(path, dates):
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["date"])
		for day in sorted(dates):
			writer.writerow([day.isoformat()])


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def run_missing_date_checker():
	if RESOLUTION not in {"5500m", "750m"}:
		raise ValueError('RESOLUTION must be set to "5500m" or "750m"')
	if not INPUT_DIR.is_dir():
		raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

	zip_ranges = find_zip_ranges(INPUT_DIR)
	if not zip_ranges:
		raise RuntimeError(f"No range-named ZIP files found in {INPUT_DIR}")

	expected_dates, present_dates, zero_byte_dates = (
		collect_zip_data(zip_ranges)
	)
	missing_dates = (expected_dates - present_dates) | zero_byte_dates

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	count_missing_days(missing_dates, expected_dates)
	write_missing_matrix(expected_dates, present_dates)
	write_date_list(ZERO_BYTE_OUTPUT, zero_byte_dates)

	print(f"Resolution: {RESOLUTION}")
	print(f"Scanned ZIP folder: {INPUT_DIR}")
	print(f"ZIP files found: {len(zip_ranges)}")
	print(f"Expected dates: {len(expected_dates)}")
	print(f"Present TIFF dates: {len(present_dates & expected_dates)}")
	print(f"Zero-byte TIFF dates: {len(zero_byte_dates & expected_dates)}")
	print(f"Missing dates: {len(missing_dates)}")
	print(f"Wrote: {COUNT_OUTPUT}")
	print(f"Wrote: {MATRIX_OUTPUT}")
	print(f"Wrote: {ZERO_BYTE_OUTPUT}")
	print("Matrix CSV meaning: True=present data, False=absent data, Null=non-applicable date.")


if __name__ == "__main__":
	run_missing_date_checker()
