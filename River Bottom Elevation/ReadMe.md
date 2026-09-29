River Bottom Elevation: 

1. Downloaded SRTM DEM from opentopography.org with the following dimensions:


- Default CRS - WGS84 [EPSG: 4326]
- X_min: 77.7588960830224352
- X_max: 78.1209612946042995
- Y_min: 13.4102404711743954
- Y_max: 13.6508534282382925

2. Reproject SRTM_DEM.tif and watershed_stream_100 shapefile to CRS:32643 (project CRS). 
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
