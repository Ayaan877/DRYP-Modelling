# Surface Flux Boundary Conditions
Hourly surface water flux data and boundary location of the HN-valley discharge inlet to the Chikkaballapur watershed area. 

## Steps
1. Open the shapefiles for `Updated Watershed`, `Channels` and `Tanks` in QGIS. Use Processing Toolbox -> Vector Geometry -> Polygon to Line to transform the watershed area into a boundary line. 
2. Reproject all features to CRS: EPSG:32643 - WGS 84/UTM zone 43N (project CRS) if not already assigned. This is to ensure the coordinates are in the correct format. 
3. Use Processing Toolbox -> Vector Overlay -> Line Intersections and find the intersection of `Channels` and `Watershed Boundary` to identify the location of the HN-valley inlet. 
4. Open the attribute table of the intersection point and using the Field Calculator create two new decimal variables - x_coord and y_coord and use the expression `$x` and `$y` for them, respectively. This will give us the coordinates of the inlet in the project CRS units (easting and northing). Also edit the id to be "0" and label to be "inlet_0".
5. Create a new CSV file `HN_valley_inflow_coord.csv` and make coloumns - ID, East, North, Elevation and Label. Save the ID, label and coordinates to the csv.
6. Add the raster layer `DEM_250.asc` to the project and inspect the gridcell in which the inlet point lies. Add the elevation of that cell to the table. 
7. Run `HN_valley_hourly` to create an hourly dataset of surface flux in mm/hr/gridcell for a duration of 10 yrs with the following features: 
- Assign zero-flux for all years prior to 2020 and for the months of June-Sept (monsoon). 
- Read the total flux allocation from `HN_valley_allocation.csv` and distribute 50% of the total allocation for all tanks except for the source tank 1 (Chalakaparthi Kere) to all non-zero flux days. Note that the source data is in MLD (million litres per day) which has to be converted to mm/hr/cell (millimetres per hour per 250m gridcell). 
8. Specify the total duration (2015-2025) and the discharge start date (01/01/2022) and dry months (June-Sept) before running the script. The output will be exported as `HN_valley_hourly_inflow.csv`. 

## Inputs
- `HN_valley_allocation.csv` - Tank-wise water allocation in MLD
- `Channels` and `Tanks` shapefiles and sidecars for the tank cluster network
- `Updated Watershed` shapefile and sidecars

## Output Format
- Names: `HN_valley_inflow_coord.csv`, `HN_valley_hourly_inflow.csv`
- Format: Comma Seperated Values (.csv)
- Duration: 01-01-2015 00:00:00 to 31-12-2025 23:59:59
- Time dim: 96432 (hours from 01-01-2015 00:00:00)
- CRS: EPSG:32643 - WGS 84/UTM zone 43N
- Inlet x-coord: 804592.40511458809487522 E
- Inlet y-coord: 1493711.53192507312633097 N
- Inlet elevation: 907.669677734375 m 
- Flux units: mm/hr/250m cell