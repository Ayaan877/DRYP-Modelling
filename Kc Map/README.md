# Crop Coefficient Map
Daily crop coefficient (Kc) datasets and a spatiotemporal crop coefficient (Kc) netCDF file for the Chikkaballapur watershed area.

## Steps
1. Create `Kc_FAO56.csv` from [FAO-56: Chapter 6 - Single crop coefficient (Kc)](https://www.fao.org/4/x0490e/x0490e0b.htm) for all crops within the crop map classes. For all crops that are not directly present, choose appropriate proxy crops. This file should contain classification, season, crop Name, FAO proxy name, growth period (min and max), growth stage durations (ini, dev, mid and late) and crop coefficient (Kc_ini, Kc_mid, Kc_end). 
2. Also add sowing period (determined from field workers/farmer interviews) and seasonal area for Kharif, Rabi and Summer (from Belle survey) to the same input metadata. 
3. Run `clean_Kc_metadata.py` to clean the FAO-56 crop metadata into `cropwise_Kc.csv`. This converts sowing periods into a single sowing date at the beginning of the sowing period, and identifies the max seasonal area of each crop, needed for weighting the class-wise curves later.
4. Run `make_cropwise_Kc_curves.py` to convert crop metadata into multi-year crop-wise Kc curves with daily temporal resolution. Input the time origin, length of dataset and fallow land Kc (here I assume 0.15) before running.
5. Run `make_classwise_Kc_curves.py` to take the area-weighted average of the Kc curves for crops within each class to construct class-wise Kc curves. Each crop is weighted by the max seasonal area for that crop across all 3 seasons. Also create a new intercropping class 7, which is weighted as $1:7 :: \text{class 4}:\text{finger millet}$.  
7. Run `make_spatiotemporal_Kc.py` to combines Kharif and Rabi crop rasters, class-wise curves, and the watershed boundary to create a daily 10 m resolution NetCDF Kc dataset for the entire 10 yr duration. When classes overlap between the two maps, assign whichever class has a higher Kc value at that time. For the overlap between class 2 (finger millet/maize) and class 4 (avarekalu/horse gram) assign it the new intercropping class 7 Kc curve. Crop the final map with the watershed boundary by assigning all gridcells outside the boundary as nodata (-99999).
8. Run `downscale_kc_netcdf.py` to regrid the 10 m NetCDF Kc dataset to a fixed 250 m grid that matches the project CRS and dimensions. 
9. Run `exclude_soil_types_kc.py` to exclude habitation, dykes and ridges, rocky outcrops, and water bodies from the final Kc map by masking these raster areas with nodata values (-99999). Export the final output as `Kc_spatiotemporal_v3.1_250m.nc`. 

Optional:
- Run `classwise_calendar.py` to creates a class-wise calendar with crop composition, union growth periods, and annual Kc ranges. This is just a summary of the classwise Kc-curves. 
- Use `plot_Kc.py` to view both the crop-wise and class-wise Kc curves.
- Run `calculate_effective_area.py` to determine the effective area of each crop class in the final output, after overlap conflicts are resolved (note that this is not necessarily equal to the crop area in the crop-map geoTIFFs). 

## Inputs
- `Kc_FAO56.csv` - Crop-wise metadata
    - Classification
    - Season
    - Crop Name
    - FAO Proxy Name
    - Sowing Period
    - Seasonal Area (for Kharif, Rabi and Summer)
    - Growth Period (min and max)
    - Growth Stage Durations (ini, dev, mid and late)
    - Crop Coefficient (Kc_ini, Kc_mid, Kc_end)
- Kharif and Rabi crop maps geoTIFFs - `CropMap_Kharif_2024-25_new.tif` and `CropMap_Rabi_2024-25_new.tif`
- `Updated Watershed` shapefile and sidecars for watershed area boundary. 
- `Soil_Texture.tif` soil texture GEOtiff 

## Output Details
- Name: `Kc_spatiotemporal_v3.1_250m.nc`
- Format: netCDF (.nc)
- Duration: Jan 2015 - Dec 2025 
- Time dim: 3653 (days since 01-01-2015 00:00:00) 
- CRS: EPSG:32643 - WGS 84 / UTM zone 43N
- Lat extent: 798778.8125 - 837778.8125
- Lon extent: 1484563.25 - 1510813.25
- Pixel size: 250 x 250 m
- Kc units: dimless
- lat units: UTM easting, m
- lon units: UTM northing, m 

## Kc Version History
v1.0: 2 yr dataset, Perennials grow annually \
v2.0: 10 yr dataset, Perennials grow consecutively. Added downscaled version 10m -> 250m \
v3.0: Outside watershed boundary → nodata \
v3.1: Excluded habitation, rocky outcrops, water bodies and dyke and ridges -> nodata
