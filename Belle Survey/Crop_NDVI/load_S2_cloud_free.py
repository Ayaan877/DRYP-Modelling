import csv, json
from datetime import datetime
import ee
from ee_plugin import Map
from qgis.core import (QgsProject, QgsGeometry,
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform)

# ---- EDIT THESE ----
WATERSHED_LAYER = 'updated_watershed'
DATES_CSV = 'C:/Users/Ayaan Dutt/Desktop/WELL Labs/DRYP Modelling/Remote Sensing/Sentinel2_L2_cloud_lte30.csv'
DATE_FORMAT = '%d-%m-%Y'      # how dates are written in the CSV, e.g. '%d/%m/%Y' for 28/10/2026
SEASON_START = '01-01-2022'   # first day of the season-year you are checking
SEASON_END = '01-03-2022'     # last day of the season-year
MAX_CLOUD = 30                # ignore CSV rows with cloud cover above this (%)
IMAGE_STYLE = 'agriculture_swir'    # 'true_color', 'color_infrared', or 'agriculture_swir'
VISIBLE_LAYERS = 0            # how many layers to switch on at load; keep small to avoid GEE timeouts
# --------------------

VISUALIZATIONS = {
    'true_color': {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.2,
    },
    'color_infrared': {
        'bands': ['B8', 'B4', 'B3'],
        'min': 0,
        'max': 4000,
        'gamma': 1.2,
    },
    'agriculture_swir': {
        'bands': ['B11', 'B8', 'B2'],
        'min': 0,
        'max': 4000,
        'gamma': 1.2,
    },
}

# 1. Read the dates from the CSV and keep those inside the season
season_start = datetime.strptime(SEASON_START, DATE_FORMAT).date()
season_end = datetime.strptime(SEASON_END, DATE_FORMAT).date()
dates = set()
with open(DATES_CSV, newline='', encoding='utf-8-sig') as fh:
    for row in csv.DictReader(fh):
        date = datetime.strptime(row['date'].strip(), DATE_FORMAT).date()
        cc = float(str(row['cloud_cover_percent']).replace('%', '').strip())
        if season_start <= date <= season_end and cc <= MAX_CLOUD:
            dates.add(date.isoformat())
DATES = sorted(dates)
print(len(DATES), 'dates in this season:', DATES)

# 2. Turn the watershed into an Earth Engine area
ws = QgsProject.instance().mapLayersByName(WATERSHED_LAYER)[0]
tr = QgsCoordinateTransform(ws.crs(), QgsCoordinateReferenceSystem('EPSG:4326'), QgsProject.instance())
geom = QgsGeometry.unaryUnion([f.geometry() for f in ws.getFeatures()])
geom.transform(tr)
aoi = ee.Geometry(json.loads(geom.asJson()))

# 3. Show one false-colour layer per date, cut to the watershed
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
if IMAGE_STYLE not in VISUALIZATIONS:
    raise ValueError(
        f"Unknown IMAGE_STYLE {IMAGE_STYLE!r}; choose from {', '.join(VISUALIZATIONS)}"
    )

vis = VISUALIZATIONS[IMAGE_STYLE]
for i, d in enumerate(DATES):
    # Hidden layers send no tile requests until ticked in QGIS
    img = s2.filterDate(d, ee.Date(d).advance(1, 'day')).select(vis['bands']).mosaic()
    Map.addLayer(img, vis, f'S2 {IMAGE_STYLE} {d}', shown=i < VISIBLE_LAYERS)
print('Done')