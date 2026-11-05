from pathlib import Path
import cdsapi

dataset = "sis-agrometeorological-indicators"
years = [str(year) for year in range(2015, 2026)]
request = {
    "variable": "reference_evapotranspiration",
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "version": "2_0",
    "area": [14, 77.5, 13, 78.5]
}

client = cdsapi.Client()
output_dir = Path(__file__).resolve().parent / "agERA5 ref ET"
output_dir.mkdir(parents=True, exist_ok=True)
for year in years:
    year_request = {**request, "year": [year]}
    output_path = output_dir / f"agera5_refET_{year}.zip"
    client.retrieve(dataset, year_request, output_path)
    print("\nDownloaded data for year:", year)