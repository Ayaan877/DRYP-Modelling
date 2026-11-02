from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CROP_MAP_DIR = SCRIPT_DIR / "Crop_Map_GPKGs"
TARGET_YEAR = 2022
TARGET_SEASON = "kharif"
ASSIGNMENT_LAYER = "crop_assignments"
OUTPUT_DIR = SCRIPT_DIR / "Crop_Matrices"

EXCLUDED_CROP_NAMES = {"fallow land", "non-agricultural land"}
EXCLUDED_CROP_CATEGORIES = {"fallow / non-agricultural"}


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def build_input_path(year, season):
	return CROP_MAP_DIR / f"{year}_{season}_belle_survey_crop_map.gpkg"


def build_output_path(year, season):
	return OUTPUT_DIR / f"{year}_{season}_crop_cooccurrence_matrix.csv"


def load_assignments(gpkg_file, layer):
	# GPKG files are SQLite databases; reading the attribute table directly skips geometry.
	if not gpkg_file.exists():
		raise FileNotFoundError(f"Missing crop map file: {gpkg_file}")
	connection = sqlite3.connect(gpkg_file)
	try:
		query = f'SELECT spatial_fid, crop_name, crop_category FROM "{layer}"'
		return pd.read_sql_query(query, connection)
	finally:
		connection.close()


def filter_agricultural_crops(assignments):
	crop_name = assignments["crop_name"].astype("string").str.strip()
	crop_category = assignments["crop_category"].astype("string").str.strip().str.lower()
	is_excluded = (
		crop_name.str.lower().isin(EXCLUDED_CROP_NAMES)
		| crop_category.isin(EXCLUDED_CROP_CATEGORIES)
	).fillna(False)
	keep = crop_name.notna() & assignments["spatial_fid"].notna() & ~is_excluded
	filtered = assignments.loc[keep, ["spatial_fid", "crop_name"]].copy()
	filtered["crop_name"] = crop_name.loc[keep]
	return filtered


def build_cooccurrence_matrix(assignments):
	"""Off-diagonal: polygons containing both crops. Diagonal: total assignment rows for the crop."""
	crops = sorted(assignments["crop_name"].unique())
	crop_index = {crop: idx for idx, crop in enumerate(crops)}

	# One row per polygon-crop pair -> polygon x crop presence matrix
	presence = (
		assignments.drop_duplicates(["spatial_fid", "crop_name"])
		.assign(present=1)
		.pivot(index="spatial_fid", columns="crop_name", values="present")
		.reindex(columns=crops)
		.fillna(0)
		.to_numpy(dtype=np.int64)
	)
	matrix = presence.T @ presence

	occurrences = assignments["crop_name"].value_counts()
	for crop, count in occurrences.items():
		matrix[crop_index[crop], crop_index[crop]] = count

	return pd.DataFrame(matrix, index=crops, columns=crops)


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	season = str(TARGET_SEASON).strip().lower()
	input_file = build_input_path(TARGET_YEAR, season)
	output_file = build_output_path(TARGET_YEAR, season)

	print(f"Loading {input_file.name}...")
	assignments = load_assignments(input_file, ASSIGNMENT_LAYER)
	assignments = filter_agricultural_crops(assignments)
	if assignments.empty:
		raise ValueError(f"No agricultural crop records found in {input_file.name}")

	matrix = build_cooccurrence_matrix(assignments)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	matrix.to_csv(output_file, encoding="utf-8-sig")
	print(f"Crops in matrix: {len(matrix):,}")
	print(f"Wrote {output_file}")
