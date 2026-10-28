from pathlib import Path
import json
import re
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_SOURCES = [
	{
		"input_dir": SCRIPT_DIR / "Crop_Data_Kharif-Rabi",
		"pattern": "*_kharif-rabi_crop_data.csv",
	},
	{
		"input_dir": SCRIPT_DIR / "Crop_Data_Summer",
		"pattern": "*_summer_crop_data.csv",
	},
]
CHUNK_SIZE = 200_000
OUTPUT_ENCODING = "utf-8-sig"

KANNADA_COLUMN = "Crop Detail"
DETAIL_ENGLISH_COLUMN = "Crop Detail (English)"
CROP_NAME_COLUMN = "Crop Name (Standardised)"
CROP_CATEGORY_COLUMN = "Crop Category"
KANNADA_COLUMN_ALIASES = ["Crop Detail", "Crop"]

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
	input_files = []
	for source in INPUT_SOURCES:
		matched_files = sorted(source["input_dir"].glob(source["pattern"]))
		if not matched_files:
			raise FileNotFoundError(
				"No files matched "
				f"{source['pattern']!r} in {source['input_dir']}"
			)
		input_files.extend(matched_files)

	input_files = sorted(input_files)
	if not input_files:
		raise FileNotFoundError("No input files found.")
	return input_files


def normalize_header_name(header):
	return str(header).replace("\ufeff", "").strip().lower()


def get_column_lookup(input_file):
	header = pd.read_csv(input_file, nrows=0)
	return {normalize_header_name(column): column for column in header.columns}


def resolve_column(column_lookup, aliases):
	for alias in aliases:
		lookup_key = normalize_header_name(alias)
		if lookup_key in column_lookup:
			return column_lookup[lookup_key]

	raise ValueError(f"Missing required columns. Tried aliases: {aliases}")


def is_kannada_text(value):
	text = normalize_text(value)
	if text is None:
		return False
	return bool(KANNADA_CHAR_PATTERN.search(text))


def build_crop_mapping(input_files):
	canonical = {}

	for input_file in input_files:
		column_lookup = get_column_lookup(input_file)
		kannada_column = resolve_column(column_lookup, KANNADA_COLUMN_ALIASES)
		detail_english_column = resolve_column(column_lookup, [DETAIL_ENGLISH_COLUMN])
		crop_name_column = resolve_column(column_lookup, [CROP_NAME_COLUMN])
		crop_category_column = resolve_column(column_lookup, [CROP_CATEGORY_COLUMN])
		use_columns = [
			kannada_column,
			detail_english_column,
			crop_name_column,
			crop_category_column,
		]

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
				kannada_name = normalize_text(row_data[kannada_column])
				if not is_kannada_text(kannada_name):
					continue

				english_detail = normalize_text(row_data[detail_english_column])
				crop_name = normalize_text(row_data[crop_name_column])
				crop_category = normalize_text(row_data[crop_category_column])

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


def summarize_canonical_mapping(canonical):
	canonical_frame = pd.DataFrame.from_dict(canonical, orient="index")
	standardized_crop_name_count = canonical_frame["crop_name"].nunique(dropna=True)
	category_count = canonical_frame["crop_category"].nunique(dropna=True)
	subcategory_count = canonical_frame[["crop_name", "crop_details"]].drop_duplicates().shape[0]

	return (
		int(standardized_crop_name_count),
		int(category_count),
		int(subcategory_count),
	)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	input_files = read_input_files()
	canonical = build_crop_mapping(input_files)
	write_outputs(canonical)
	(
		standardized_crop_name_count,
		category_count,
		subcategory_count,
	) = summarize_canonical_mapping(canonical)

	print(f"Input files: {len(input_files)}")
	print(f"Kannada crop keys saved: {len(canonical):,}")
	print(f"Standardized crop names: {standardized_crop_name_count:,}")
	print(f"Crop categories: {category_count:,}")
	print(f"Subcategories (crop details per crop name): {subcategory_count:,}")
	print(f"Kannada crop dictionary: {OUTPUT_DICT.name}")
	print(f"Kannada crop table: {OUTPUT_KEY.name}")
