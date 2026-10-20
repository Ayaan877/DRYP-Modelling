# Rootzone Soil Moisture
Rootzone Soil Moisture data extracted from https://zenodo.org/records/17014507 for the Chikkaballapur watershed area.

## Steps
1. Download `RF_predcited_RZSM_1981_2024_netcdef.zip` from source. This contains a netCDF file for India with (lat, long) given in decimal deg north/east, and RZSM given in volume of water/volume of soil within 0-100 cm depth. The data ranges from Jan 1981 to Dec 2024, with daily temporal resolution.
2. Transform the `Updated Watershed` to the same coordinate system (EPSG:4326 - WGS 84) and overlay it on the netCDF. 
3. Input a date range and output file name. Preserve all grid cells that contain the boundary of the watershed, such that the entire area within the watershed has an associated RZSM value. 
4. Reproject the extracted soil moisture netCDF to EPSG:32643 while retaining
the source grid dimensions. This is saved as
`RZSM_2014-2024_5500m.nc`.
5. Resample the reprojected file to the 250 m project grid using
`downscale_netcdf.py`. This is saved as `RZSM_2014-2024_250m.nc`.

## Inputs
- `RF_predcited_RZSM_1981_2024.nc`
- `Updated Watershed` shapefile

## Output Details
- Names: `RZSM_2014-2024_5500m.nc`, `RZSM_2014-2024_250m.nc`
- Format: netCDF (.nc)
- Duration: Jan 2014 - Dec 2024 
- Time dim: 4018 (days since 01-01-2014 00:00:00) 
- CRS: EPSG:32643 - WGS 84 / UTM zone 43N
- Lat extent: 798778.8125 - 837778.8125
- Lon extent: 1484563.25 - 1510813.25
- Pixel size: approximately 5500 m for the first file; 250 x 250 m for the
  `_250m` file
- RZSM units: m3/m3
- lat units: UTM easting, m
- lon units: UTM northing, m 