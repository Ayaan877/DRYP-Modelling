from pathlib import Path
import geopandas as gpd
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CROP_MAP_DIR = SCRIPT_DIR / "Crop_Map_GPKGs"
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
OUT_DIR = SCRIPT_DIR / "crop_type_analysis"
OUTPUT = OUT_DIR / "watershed_crop_acreage_by_season.csv"

ASSIGNMENT_LAYER = "crop_assignments"
SEASON_ORDER = {"kharif": 0, "rabi": 1, "summer": 2}  # chronological within a year
NON_CROP_CATEGORY = "Fallow / Non-Agricultural"
INPUT_COLUMNS = [
    "spatial_fid",
    "survey_area_acres",
    "cropped_area_acres",
    "crop_name",
    "crop_category",
]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_watershed():
    watershed = gpd.read_file(WATERSHED_FILE)
    if watershed.empty:
        raise ValueError(f"No watershed geometry found in {WATERSHED_FILE}")
    return watershed


def read_watershed_assignments(path, watershed):
    """Crop assignment rows whose survey polygon intersects the watershed."""
    layer_crs = gpd.read_file(path, layer=ASSIGNMENT_LAYER, rows=1).crs
    mask = watershed.to_crs(layer_crs).union_all()
    assignments = gpd.read_file(
        path, layer=ASSIGNMENT_LAYER, columns=INPUT_COLUMNS, mask=mask
    )
    assignments["crop_name"] = assignments["crop_name"].astype("string").str.strip()
    # Fallow / non-agricultural rows carry no cropped area, so use the polygon area
    non_crop = assignments["crop_category"].eq(NON_CROP_CATEGORY)
    assignments["area_acres"] = assignments["cropped_area_acres"].where(
        ~non_crop | assignments["cropped_area_acres"].gt(0),
        assignments["survey_area_acres"],
    )
    return assignments


def watershed_crop_acreage_pivot():
    """Write crop acreage inside the watershed with crop names as rows and seasons as columns."""
    watershed = load_watershed()
    map_files = sorted(CROP_MAP_DIR.glob("*_belle_survey_crop_map.gpkg"))
    if not map_files:
        raise FileNotFoundError(f"No crop-map GeoPackages found in {CROP_MAP_DIR}")

    totals = {}
    for path in map_files:
        year, season = path.name.split("_")[:2]
        label = f"{year} {season.capitalize()}"
        assignments = read_watershed_assignments(path, watershed)
        totals[(int(year), SEASON_ORDER[season], label)] = (
            assignments.groupby("crop_name")["area_acres"].sum()
        )
        print(
            f"{label}: {assignments['spatial_fid'].nunique():,} polygons, "
            f"{assignments['crop_name'].nunique()} crop names"
        )

    columns = [key[2] for key in sorted(totals)]
    pivot = (
        pd.DataFrame({key[2]: series for key, series in totals.items()})
        .fillna(0)
        .reindex(columns=columns)
    )
    pivot["total_area_acres"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_area_acres", ascending=False)
    pivot.index.name = "crop_name"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT)
    print(f"Saved {OUTPUT} ({len(pivot)} crop names)")

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    watershed_crop_acreage_pivot()
