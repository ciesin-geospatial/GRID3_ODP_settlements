# Geospatial Data Analysis Toolset for Settlement Identification

---

***Table of contents***

1. Data Collection (OSM)
2. Feature Engineering (Land Cover, Land Use, Road Connectivity, Dispersion, Building Features)
3. False Positive Classification with Machine Learning

---

The toolset presented in this repository aims to predict the possibility that a potential settlement is a true positive (an actual settlement) or a false positive (a non-settlement area that is mistaken as a settlement during the generation stage), in an attempt to improve the accuracy of settlement identification work. 

The generation of candidate settlements is based on image analysis of high-resolution satellite imagery data, which will not be covered in this repo. The toolset here will take in potential settlements and help predict which ones are likely to be false positives. False negatives (actual settlements that are missed in the generation stage) will not be discovered, so it is preferable to be more lenient during the generation process and allow more candidates to be considered in this subsequent filtering stage. It is worth noting that this toolset is developed with small settlements (hamlet or village level) in mind, taking advantage of their relatively smaller scope and simpler geometric shape. However, components of the toolset can be transferred to the filtering of larger settlements. 

***List of utility functions***

---
Function `read_csv_as_gpd`

Parameters: 
- `df_or_filepath`, *GeoDataframe or str*
- `id_column`, *str*   (default = None, if not specified, a UUID column will be generated) 
- `lon_lat_columns`, *list of str*   (default = \[\]) 
- `attribute_columns`, *list of str*  (default = \[\]) 
- `drop_rows_where_duplicates_in_columns`, *list of str*  (default = \[\]) 
- `keep_which_if_duplicates`, *str*  (default = 'first', options: 'first','last')
- `drop_rows_where_nan_in_columns`, *list of str*  (default = \[\]) 

Returns: 
- A GeoDataframe loaded from a CSV file. Use `drop_rows_where_duplicates_in_columns` in combination with `id_column` and `keep_which_if_duplicates` to keep control the behavior when duplicates are detected in certain columns. Use `drop_rows_where_nan_in_columns` to control the dropping of rows with missing values.

---
Function `explode_geometry`

Parameters: 
- `gdf`, *GeoDataframe* 
- `geom_colum`, *str*   (default = 'geometry') 
- `id_column`, *str*   (default = None, must specify if using drop_duplicates) 
- `drop_duplicates`, *boolean*   (default = False) 

Returns: 
- A GeoDataframe with geometry column exploded into single-polygon/single-linestring objects. Use drop_duplicates in combination with id_column to keep only the the largest shape among the shapes with the same original id.

---
Function `left_spatial_join`

Parameters: 
- `gdf1`, *GeoDataframe*
- `gdf2`, *GeoDataframe*   

Returns: 
- A GeoDataframe resulting from the left spatial join of two input GeoDataframes. Spatial join operation uses "intersection", right index dropped.

---
Function `drop_bounds`

Parameters: 
- `gdf`, *GeoDataframe* 
- `geom_column`, *str* 

Returns: 
- A GeoDataframe with geometry bounds columns dropped, these include the columns that start with a geometry column name and end with `minx`, `maxx`, `miny`, or `maxy`. This is a utility function to help similify output of `get_raster_value_distribution` function.

---
Function `add_buffer_column`

Parameters: 
- `gdf`, *GeoDataframe* 
- `buffer_radius`, *int*  (in meters)
- `geom_column`, *str*   (default = 'geometry') 
- `buffer_shape`, *str*   (default = 'round', options: 'round','square','flat') 
- `proj2`, *str* (default = None, options: any valid crs)
- `replace`, *boolean* (default = False)
- `return_new_column`, *boolean* (default = False)

Returns: 
- A GeoDataframe with a new buffer column. Set the geometry from which to buffer using `geom_column`. Set buffer radius with `buffer_radius`, units are in meters, the radius can be negative for shrinking shapes, though multiple shapes may be created as a result. Control shape of buffer with `buffer_shape`. If you want the buffer to be based on a projection other than current projection, use `proj2`, it will not affect the projection of the input dataframe, it only applies to the new buffer column. If you want to replace the main geometry column with the newly created buffer column, set `replace` to True. If you want to get the name of the newly created buffer column, set `return_new_column` to True. 

---
Function `add_centroid_column`

Parameters: 
- `gdf`, *GeoDataframe* 
- `geom_column`, *str*   (default = 'geometry') 
- `proj2`, *str* (default = None, options: any valid crs)
- `replace`, *boolean* (default = False)
- `return_new_column`, *boolean* (default = False)

Returns: 
- A GeoDataframe with a new centroid column. Set the geometry from which to calculate centroid using `geom_column`. If you want the centroid to be based on a projection other than current projection, use `proj2`, it will not affect the projection of the input dataframe, it only applies to the new centroid column. If you want to replace the main geometry column with the newly created centroid column, set `replace` to True. If you want to get the name of the newly created centroid column, set `return_new_column` to True. 

---
Function `add_intersection_count_column`

Parameters:
- `gdf`, *GeoDataframe* 
- `uuid_column`, *str* 
- `buffer_column`, *str* 
- `feature_layer`, *str* 
- `new_column`, *str* 
- `feature_geom_column`, *str* (default = 'geometry')

Returns: 
- A GeoDataframe with a new column that counts the feature geometries within the buffer of main geometry. `gdf` is the main GeoDataframe that has a buffer column, specified by `buffer_column`. `feature_layer` is the other GEoDataframe with features, by default the 'geometry' column of the feature layer will be used but it can be changed. `new_column` controls the name of the newly-created column.

---
Function `add_distance_to_nearest_neighbor_column`

Parameters:
- `gdf`, *GeoDataframe* 
- `geom_centroid_column`, *str* 
- `new_column`, *str* 
- `rounding`, *str* (default = 0)

Returns: 
- A GeoDataframe with a new column that calculate the distance from this geometry to the nearest geometry within the same GeoDataframe. Use `geom_centroid_column` to specify which geometry to do nearest distance calculation. `new_column` controls the name of the newly-created column. The distance is measured in meters and rounded by default, but can be changed with `rounding` parameter. 

---
Function `add_covering_geotiff_column`

Parameters: 
- `gdf`, *GeoDataframe* 
- `geom_column`, *str* 
- `geotiff_filepath_column`, *str*
- `geotiff_filepath_list`, *list of str* 

Returns: 
- A GeoDataframe with a new column storing the filepaths of the geotiff that cover the shapes in each row. Set the geometry with which to find geotiff using `geom_column`. Set the name of the new geotiff_filepath column with `geotiff_filepath_column`. Provide the filepaths of the candidate geotiffs in `geotiff_filepath_list`. All parameters need to be explicitly specified.

---
Function `get_raster_value_distribution`

Parameters: 
- `gdf`, *GeoDataframe* 
- `id_column`, *str*
- `geom_column`, *str* 
- `geotiff_filepath_column`, *str*
- `geotiff_filepath_list`, *list of str* 
- `code_to_label_mapping`, *list of str* 
- `label_marker`, *list of str* 
- `normalize`, *boolean* 

Returns: 
- A GeoDataframe with new columns corresponding to the distribution of different codes in the covering raster image. Specify which raster geotiff is covering the geometry with `geotiff_filepath_column`. Provide the filepaths of the candidate geotiffs in `geotiff_filepath_list`. Use `code_to_label_mapping` to specify the mapping from numerical codes to human-readable labels, this may vary from one standard to another. `label_marker` is a prefix to all the newly-created column, so as to mark which raster these columns are derived from. `normalize` controls whether the values in the columns are proportion or absolute count of pixels. All parameters need to be explicitly specified.

---
Function `get_groupby_stats_df`

Parameters:
- `data`, *Dataframe or GeoDataframe* 
- `groupby_column`, *str* 
- `stats_map`, *dict* 

Returns: 
- A Dataframe with statistics of the provided features, a simple wrapper around Pandas `groupby` function.

---
Function `get_most_correlated_feature`

Parameters:
- `data`, *Dataframe or GeoDataframe* 
- `target`, *str* 
- `features`, *list of str* 

Returns: 
- The name of feature that is most correlated with the target, as measured by Pearson R. This is a simple utility function for choosing one feature when several features are highly correlated with each other.

---
Function `within_value_to_range_value`

Parameters:
- `gdf`, *GeoDataframe* 
- `buffer_radius_markers`, *str* 

Returns: 
- A GeoDataframe with new columns tracking the count of features in the ring areas around main geometries. For example, the number of features in the ring area that is at least 500 meters away but at most 5000 meters away from a settlement. This is based on the observation that number of features within 5000 meters must include the number of features within 500 meters, which creates collinearity that hurts prediction models. Thus, this function calculates the count in a specific range instead of the count within a radius.

---