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

ID_COLUMN = "KGISCode"
VILLAGE_CANDIDATE_COLUMNS = ["KGIS_village_name", "Village"]
NAME_FIELDS = ["village", "district", "taluk", "hobli"]
FIELD_TO_SOURCE_COLUMN = {
	"district": "District",
	"taluk": "Taluk",
	"hobli": "Hobli",
}

OUTPUT_DICT = SCRIPT_DIR / "village_id_dict.json"
OUTPUT_KEY = SCRIPT_DIR / "village_id_canonical.csv"
OUTPUT_CONFLICTS = SCRIPT_DIR / "village_id_conflicts.csv"

ENGLISH_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\s\.\-'/&,()]*$")

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


def normalize_for_match(value):
	text = normalize_text(value)
	if text is None:
		return None
	return text.casefold()


def is_english_name(value):
	text = normalize_text(value)
	if text is None:
		return False
	return bool(ENGLISH_NAME_PATTERN.fullmatch(text))


def read_input_files():
	input_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
	if not input_files:
		raise FileNotFoundError(
			f"No files matched {INPUT_PATTERN!r} in {INPUT_DIR}"
		)
	return input_files


def ensure_entry(container, key):
	if key not in container:
		container[key] = {field: None for field in NAME_FIELDS}


def build_canonical_lookup(input_files):
	canonical = {}
	variants = {}

	use_columns = [ID_COLUMN] + VILLAGE_CANDIDATE_COLUMNS + list(
		FIELD_TO_SOURCE_COLUMN.values()
	)
	use_columns = list(dict.fromkeys(use_columns))

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
				kgis_village_id = normalize_text(row_data[ID_COLUMN])
				if kgis_village_id is None:
					continue

				ensure_entry(canonical, kgis_village_id)
				ensure_entry(variants, kgis_village_id)

				field_values = {
					"village": normalize_text(
						row_data[VILLAGE_CANDIDATE_COLUMNS[0]]
					)
					or normalize_text(row_data[VILLAGE_CANDIDATE_COLUMNS[1]]),
					"district": normalize_text(
						row_data[FIELD_TO_SOURCE_COLUMN["district"]]
					),
					"taluk": normalize_text(
						row_data[FIELD_TO_SOURCE_COLUMN["taluk"]]
					),
					"hobli": normalize_text(
						row_data[FIELD_TO_SOURCE_COLUMN["hobli"]]
					),
				}

				for field_name in NAME_FIELDS:
					field_value = field_values[field_name]
					if not is_english_name(field_value):
						continue

					if canonical[kgis_village_id][field_name] is None:
						canonical[kgis_village_id][field_name] = field_value

					normalized_spelling = normalize_for_match(field_value)
					if normalized_spelling is None:
						continue

					field_variants = variants[kgis_village_id][field_name]
					if field_variants is None:
						field_variants = {}
						variants[kgis_village_id][field_name] = field_variants

					if normalized_spelling not in field_variants:
						field_variants[normalized_spelling] = field_value

	canonical = {
		key: value
		for key, value in canonical.items()
		if any(value[field] is not None for field in NAME_FIELDS)
	}

	return canonical, variants


def build_conflict_report(canonical, variants):
	conflicts = []

	for kgis_village_id in sorted(canonical.keys()):
		for field_name in NAME_FIELDS:
			field_variants = variants.get(kgis_village_id, {}).get(field_name)
			if not field_variants:
				continue

			spellings = list(field_variants.values())
			if len(spellings) <= 1:
				continue

			conflicts.append(
				{
					"kgis_village_id": kgis_village_id,
					"field": field_name,
					"canonical_value": canonical[kgis_village_id][field_name],
					"variant_count": len(spellings),
					"spellings": ", ".join(spellings),
				}
			)

	return conflicts


def write_outputs(canonical, conflicts):
	with OUTPUT_DICT.open("w", encoding="utf-8") as file:
		json.dump(canonical, file, ensure_ascii=False, indent=2)

	canonical_rows = []
	for kgis_village_id in sorted(canonical.keys()):
		row = {"kgis_village_id": kgis_village_id}
		for field in NAME_FIELDS:
			row[field] = canonical[kgis_village_id][field]
		canonical_rows.append(row)

	pd.DataFrame(canonical_rows).to_csv(
		OUTPUT_KEY,
		index=False,
		encoding=OUTPUT_ENCODING,
	)

	pd.DataFrame(conflicts).to_csv(
		OUTPUT_CONFLICTS,
		index=False,
		encoding=OUTPUT_ENCODING,
	)


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	input_files = read_input_files()
	canonical, variants = build_canonical_lookup(input_files)
	conflicts = build_conflict_report(canonical, variants)
	write_outputs(canonical, conflicts)

	print(f"Input files: {len(input_files)}")
	print(f"Canonical IDs saved: {len(canonical):,}")
	print(f"Conflicts saved: {len(conflicts):,}")
	print(f"Canonical dictionary: {OUTPUT_DICT.name}")
	print(f"Canonical table: {OUTPUT_KEY.name}")
	print(f"Conflict report: {OUTPUT_CONFLICTS.name}")
