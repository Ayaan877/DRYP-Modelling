# Rootzone Soil Moisture
Rootzone Soil Moisture data extracted from https://zenodo.org/records/17014507 for the Chikkaballapur watershed area.

## Steps
1. Download `RF_predcited_RZSM_1981_2024_netcdef.zip` from source. This contains a netCDF file for India with (lat, long) given in decimal deg north/east, and RZSM given in volume of water/volume of soil within 0-100 cm depth. The data ranges from Jan 1981 to Dec 2024, with daily temporal resolution.
2. Transform the `Updated Watershed` to the same coordinate system (EPSG:4326 - WGS 84) and overlay it on the netCDF. 
3. Input a date range and output file name. Preserve all grid cells that contain the boundary of the watershed, such that the entire area within the watershed has an associated RZSM value. 
4. Reproject the soil moisture netCDF to the project dimensions and CRS.
5. Export as a new netCDF for the desired range and area. 

## Inputs
- `RF_predcited_RZSM_1981_2024.nc`
- `Updated Watershed` shapefile

## Output Details
- Name: `RZSM_2014-2024.nc`
- Format: netCDF (.nc)
- Duration: Jan 2014 - Dec 2024 
- Time dim: 4018 (days since 01-01-2014 00:00:00) 
- CRS: EPSG:32643 - WGS 84 / UTM zone 43N
- Lat extent: 798778.8125 - 837778.8125
- Lon extent: 1484563.25 - 1510813.25
- Pixel size: 250 x 250 m
- RZSM units: m3/m3
- lat units: UTM easting, m
- lon units: UTM northing, m 