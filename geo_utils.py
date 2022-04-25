import pandas as pd
import numpy as np
import re

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil
from tqdm import tqdm
from glob import glob

import uuid

import rasterio
import geopandas as gpd
from shapely import ops

import warnings
warnings.filterwarnings('ignore')

import pyrosm
from pyrosm import get_data as get_osm_data_as_pbf

def get_not_contained_part(new, old, id_col = 'id'):
    return new[new[id_col].isin(set(new[id_col]).difference(set(old[id_col])))].copy()


def read_csv_as_gpd(df_or_filepath, id_column = None, lon_lat_columns = [], attribute_columns = [], drop_rows_where_duplicates_in_columns = [], keep_which_if_duplicates = 'first', drop_rows_where_nan_in_columns = []):
    
    all_columns = ([] if id_column is None else [id_column]) + list(set(lon_lat_columns+attribute_columns+drop_rows_where_duplicates_in_columns+drop_rows_where_nan_in_columns))
    
    if isinstance(df_or_filepath,str):
        df = pd.read_csv(df_or_filepath, usecols=all_columns)
    elif isinstance(df_or_filepath, pd.core.frame.DataFrame): 
        df = df_or_filepath[all_columns]
    
    if id_column is None:
        df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
        id_column = 'uuid'
    
    if len(drop_rows_where_duplicates_in_columns)>0:
        pre_length = len(df)
        df = df.drop_duplicates(subset = drop_rows_where_duplicates_in_columns, keep=keep_which_if_duplicates)
        post_length = len(df)
        num_rows_dropped = post_length - pre_length
        if num_rows_dropped > 0:
            print(num_rows_dropped, 'rows dropped due to duplication in columns:',','.join(drop_rows_where_duplicates_in_columns))
    
    if len(drop_rows_where_nan_in_columns)>0:
        pre_length = len(df)
        df = df.dropna(subset = drop_rows_where_nan_in_columns)
        post_length = len(df)
        num_rows_dropped = post_length - pre_length
        if num_rows_dropped > 0:
            print(num_rows_dropped, 'rows dropped due to missing values in columns:',','.join(drop_rows_where_nan_in_columns))

    df = df[[id_column]+attribute_columns+lon_lat_columns]
    
    lon_column, lat_column = lon_lat_columns
    
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df[lon_column], df[lat_column])).set_crs('epsg:4326')
    
    return gdf

def explode_geometry(gdf, geom_colum = 'geometry', drop_duplicates = True):
    gdf = gdf.explode(geom_colum).reset_index()
    if drop_duplicates:
        gdf[geom_colum+'__area_size'] = gdf[geom_colum].area
        gdf = gdf.sort_values(geom_colum+'__area_size', ascending=False)
        gdf = gdf.drop_duplicates(subset='level_0', keep='first')
        gdf = gdf.drop([geom_colum+'__area_size'], axis=1)
    gdf = gdf.drop(['level_0','level_1'], axis=1)
    return gdf

def add_centroid_column(gdf, geom_column = 'geometry', proj2 = None, replace = False, return_new_column = False):
    
    new_column = geom_column if replace else geom_column + '_centroid'
    
    proj2 == '+proj=cea' if proj2 == 'cea' else proj2
    
    if proj2:
        gdf[new_column] = gdf[geom_column].to_crs(proj2).centroid.to_crs(gdf.crs)
    else:
        gdf[new_column] = gdf[geom_column].centroid
        
    if return_new_column:
        return gdf, new_column
    return gdf

def add_buffer_column(gdf, buffer_radius, geom_column = 'geometry', cap_style = 'round', proj2 = None, replace = False, return_new_column = False):

    radius_marker = re.sub('000$','k',str(int(buffer_radius)))
    
    new_column = geom_column if replace else geom_column + '_buffer'
    
    proj2 == '+proj=cea' if proj2 == 'cea' else proj2
    
    if proj2:
        gdf[new_column] = gdf[geom_column].to_crs(proj2).buffer(buffer_radius, cap_style=cap_style).to_crs(gdf.crs)
    else:
        gdf[new_column] = gdf[geom_column].buffer(buffer_radius, cap_style=cap_style)
        
    if return_new_column:
        return gdf, new_column
    return gdf
  
def left_spatial_join(gdf1, gdf2):
    return gpd.sjoin(gdf1, gdf2, how='left').drop(['index_right'], axis=1)
    
def add_intersection_count_column(gdf, uuid_column, buffer_column, new_layer, new_column, geom_column = 'geometry', new_layer_geom_column = 'geometry', default_right_index = 'index_right'):
    temp = gpd.sjoin(gdf.set_geometry(buffer_column), new_layer[[new_layer_geom_column]], how='left').dropna(subset=[default_right_index])
    uuid_to_intersection_count_mapping = temp.groupby(uuid_column)[default_right_index].count().to_dict()
    gdf[new_column] = gdf[uuid_column].map(uuid_to_intersection_count_mapping).fillna(0).apply(int)
    return gdf

def add_bounds_column(gdf, geom_column = 'geometry'):
    bounds_columns = gdf[geom_column].bounds
    bounds_columns.columns = [geom_column+'__'+col for col in bounds_columns.columns]
    gdf = pd.concat([gdf, bounds_columns], axis=1)
    return gdf

def add_covering_geotiff_column(gdf, geom_column, geotiff_filepath_list, geotiff_filepath_column):
    
    geotiff_bounds_list = []
    for p in geotiff_filepath_list:
        geotiff_bounds_list.append(rasterio.open(p).bounds)
        
    def _find_covering_geotiff_filepath(minx, maxx, miny, maxy, geotiff_bounds_list, geotiff_filepath_list):
        for i in range(len(geotiff_bounds_list)):
            geotiff_bounds = geotiff_bounds_list[i]
            if minx > geotiff_bounds.left and maxx < geotiff_bounds.right and miny > geotiff_bounds.bottom and maxy < geotiff_bounds.top:
                return geotiff_filepath_list[i]
        return np.nan
    
    gdf = add_bounds_column(gdf, geom_column = geom_column)
    bounds_columns = [geom_column+'__'+col for col in ['minx', 'maxx', 'miny', 'maxy']]
    gdf[geotiff_filepath_column] = gdf[bounds_columns].apply(lambda row: _find_covering_geotiff_filepath(*row, geotiff_bounds_list, geotiff_filepath_list), axis=1)
    
    unmatched_count = gdf[geotiff_filepath_column].isnull().sum()
    if unmatched_count>0:
        print(unmatched_count,'rows does not have a matched geotiff.')
    
    return gdf
    
def get_raster_point_value(gdf, geom_column, raster_filepath, new_column):
    raster_file = rasterio.open(raster_filepath)
    raster = raster_file.read(1)
    gdf[new_column] = gdf[geom_column].apply(lambda point: raster_file.index(*[num[0] for num in point.xy])).apply(lambda pos: raster[pos[0], pos[1]])
    return gdf


def get_raster_value_distribution(gdf, geom_column, uuid_column, geotiff_filepath_column, geotiff_filepath_list, code_to_label_mapping, label_marker):
    
    data_rows = []
    
    for p in tqdm(geotiff_filepath_list):

        raster_file = rasterio.open(p)
        raster = raster_file.read(1)

        part = gdf.loc[gdf[geotiff_filepath_column]==p].reset_index()

        for i, row in part.iterrows():

            uuid = row[uuid_column]
            miny, minx = raster_file.index(row[geom_column+'__minx'], row[geom_column+'__miny'])
            maxy, maxx = raster_file.index(row[geom_column+'__maxx'], row[geom_column+'__maxy'])    

            # In geospatial coordinates system (lon lat) going north/up is increasing; 
            # In screen canvas, going down is increasing;
            clip = raster[maxy:miny, minx:maxx] 

            code_count_pairs = np.unique(clip, return_counts=True)
            
            code_count_mapping = dict(zip(*code_count_pairs))
            label_count_mapping = {v:code_count_mapping.get(k,0) for k,v in code_to_label_mapping.items()}
            
            data_rows.append((uuid, label_count_mapping))
    
    data_df = pd.DataFrame(data_rows, columns=[uuid_column,'label_count_mapping'])
    label_count_df = pd.json_normalize(data_df['label_count_mapping'])
    label_count_df.columns = label_marker+'__'+label_count_df.columns
    label_count_df[uuid_column] = data_df[uuid_column]
    gdf = gdf.merge(label_count_df, on=uuid_column, how='left')
    return gdf

def drop_bounds_columns(gdf, geom_column='geometry'):
    gdf = gdf.drop([geom_column+'__'+col for col in ['minx', 'maxx', 'miny', 'maxy']], axis=1)
    return gdf