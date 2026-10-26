from pathlib import Path
import json
import re
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Crop_Data_Kharif-Rabi"
INPUT_PATTERN = "*_kharif-rabi_crop_data.csv"
CHUNK_SIZE = 200_000
OUTPUT_ENCODING = "utf-8-sig"

KANNADA_COLUMN = "Crop Detail"
DETAIL_ENGLISH_COLUMN = "Crop Detail (English)"
CROP_NAME_COLUMN = "Crop Name (Standardised)"
CROP_CATEGORY_COLUMN = "Crop Category"

OUTPUT_DICT = SCRIPT_DIR / "crop_dict.json"
OUTPUT_KEY = SCRIPT_DIR / "crop_dict_canonical.csv"

KANNADA_CHAR_PATTERN = re.compile(r"[\u0C80-\u0CFF]")

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_text(value):
	if value is None or pd.isna(value):
		return None

	text = str(value).replace("\ufeff", "")
	text = re.sub(r"\s+", " ", text).strip()
	if text == "":
		return None
	return text


def read_input_files():
	input_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
	if not input_files:
		raise FileNotFoundError(
			f"No files matched {INPUT_PATTERN!r} in {INPUT_DIR}"
		)
	return input_files


def is_kannada_text(value):
	text = normalize_text(value)
	if text is None:
		return False
	return bool(KANNADA_CHAR_PATTERN.search(text))


def build_crop_mapping(input_files):
	canonical = {}

	use_columns = [
		KANNADA_COLUMN,
		DETAIL_ENGLISH_COLUMN,
		CROP_NAME_COLUMN,
		CROP_CATEGORY_COLUMN,
	]

	for input_file in input_files:
		reader = pd.read_csv(
			input_file,
			usecols=use_columns,
			dtype="string",
			chunksize=CHUNK_SIZE,
		)
		print(f"Processing {input_file.name}...")

		for chunk in reader:
			for row in chunk.itertuples(index=False, name=None):
				row_data = {
					column_name: row[index]
					for index, column_name in enumerate(chunk.columns)
				}
				kannada_name = normalize_text(row_data[KANNADA_COLUMN])
				if not is_kannada_text(kannada_name):
					continue

				english_detail = normalize_text(row_data[DETAIL_ENGLISH_COLUMN])
				crop_name = normalize_text(row_data[CROP_NAME_COLUMN])
				crop_category = normalize_text(row_data[CROP_CATEGORY_COLUMN])

				if (
					english_detail is None
					or crop_name is None
					or crop_category is None
				):
					continue

				if kannada_name in canonical:
					continue

				canonical[kannada_name] = {
					"crop_details": english_detail,
					"crop_name": crop_name,
					"crop_category": crop_category,
				}

	return canonical


def write_outputs(canonical):
	with OUTPUT_DICT.open("w", encoding="utf-8") as file:
		json.dump(canonical, file, ensure_ascii=False, indent=2)

	canonical_rows = []
	for kannada_crop_key in sorted(canonical.keys()):
		canonical_rows.append(
			{
				"kannada_crop_key": kannada_crop_key,
				"crop_details": canonical[kannada_crop_key]["crop_details"],
				"crop_name": canonical[kannada_crop_key]["crop_name"],
				"crop_category": canonical[kannada_crop_key]["crop_category"],
			}
		)

	pd.DataFrame(canonical_rows).to_csv(
		OUTPUT_KEY,
		index=False,
		encoding=OUTPUT_ENCODING,
	)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	input_files = read_input_files()
	canonical = build_crop_mapping(input_files)
	write_outputs(canonical)

	print(f"Input files: {len(input_files)}")
	print(f"Kannada crop keys saved: {len(canonical):,}")
	print(f"Kannada crop dictionary: {OUTPUT_DICT.name}")
	print(f"Kannada crop table: {OUTPUT_KEY.name}")
