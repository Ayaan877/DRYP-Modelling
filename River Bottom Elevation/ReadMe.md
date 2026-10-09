# River Bottom Elevation: 
River depth elevation of Chikkaballapur watershed area derived from SRTM DEM (from [opentopography.org](https://portal.opentopography.org/raster)) with 2m depth subtracted from river areas to indicate river depth.

## Steps
1. Downloaded SRTM DEM from [opentopography.org](https://portal.opentopography.org/raster) with the following dimensions:
- Default CRS - WGS84 [EPSG: 4326]
- X_min: 77.7588960830224352
- X_max: 78.1209612946042995
- Y_min: 13.4102404711743954
- Y_max: 13.6508534282382925
2. Reproject SRTM_DEM.tif and watershed_stream_100 shapefile back to CRS: EPSG:32643 - WGS 84/UTM zone 43N (project CRS) using processing toolbox -> raster tools -> warp (reproject). 
3. Use "fill sinks (Wang & Liu)" to repair SRTM_DEM and export as ASC. 
4. Rasterize river mask using the following parameters: 
- Data type: Unassigned
- Burn-in value: 1
- Extent: SRTM_DEM.asc
- Units: Georeferenced
- Dimensions: 30x30
5. Use r.null to change nodata values in river mask to 0. 
6. In raster calculator, use the following expression to create river bottom elevation:

if("River_Mask@1" = 1, "SRTM_DEM@1" - 2, "SRTM_DEM@1")

## Inputs
- `SRTM_DEM_Raw.tif` downloaded from [opentopography.org](https://portal.opentopography.org/raster)
- `River_Mask` shapefile and sidecars

## Output Details
- Name: `River_Bottom_Elevation.asc` (-2m at rivers), `SRTM_DEM.asc` (plain DEM)
- Format: Esri ASCII (.asc)
- CRS: EPSG:32643 - WGS 84/UTM zone 43N
- Lat extent: 798769.8778999999631196 - 837796.4396999999880791
- Lon extent: 1484563.2962543477769941 - 1510814.2519135093316436
- Dimensions: 1288 x 867
- Pixel size: 30 x 30 m
- Nodata value: -99999
- Elevation units: m
- lat units: UTM easting, m
- lon units: UTM northing, m 