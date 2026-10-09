# Rooting Depth
Crop rooting depth map for Chikkaballapur watershed area derived from crop maps and FAO-56 rooting depth data.

## Steps
1. Identify proxy crops for all crops in crop map and make `Rooting Depth FAO56.csv` from the rooting depth data in [FAO-56: Chapter 8 - ETc under soil water stress conditions](https://www.fao.org/4/x0490e/x0490e0e.htm#chapter%208%20%20%20etc%20under%20soil%20water%20stress%20conditions) with class, season, crop name, crop proxy, and rooting depth (min-max in mm).
2. Run `make_rooting_depth.py` to read rooting-depth lookup table, Kharif and Rabi crop rasters, and the watershed boundary. Use the following rules to resolve multiple RD values:
- For crops with a RD range, assign the max RD.
- For classes with multiple crops, assign the value of the crop with the max RD.
- For areas with overlapping classes (after overlaying Kharif and Rabi maps), assign the value of the class with the max RD. 
3. Save the resulting spatial file as a 10 x 10 m resolution ESRI ASCII file.
4. Use `downscale_rooting_depth.py` to regrid the 10 m rooting-depth grid to the fixed 250 m target grid with the CRS and dimensions of the project. 

## Inputs

- `Rooting Depth FAO56.csv`
- Kharif and Rabi crop-map rasters - `CropMap_Kharif_2024-25_new.tif`, `CropMap_Rabi_2024-25_new.tif`
- `Updated Watershed` shapefile and sidecars

## Output Details
- Name: `Rooting_Depth_v3.0_250m.asc`
- Format: Esri ASCII (.asc)
- CRS: EPSG:32643 - WGS 84/UTM zone 43N
- Lat extent: 798778.8010473210597411 - 837778.8010473210597411
- Lon extent: 1484563.2962543477769941 - 1510813.2962543477769941
- Dimensions: 156 x 105
- Pixel size: 250 x 250 m
- Nodata value: -99999
- Rooting Depth units: mm
- lat units: UTM easting, m
- lon units: UTM northing, m 