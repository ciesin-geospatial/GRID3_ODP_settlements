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

import scipy.spatial as scipy_spatial

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

def explode_geometry(gdf, geom_colum = 'geometry', id_column = None, drop_duplicates = False):
    gdf = gdf.explode(geom_colum).reset_index(drop=True)
    if drop_duplicates:
        if id_column is None:
            raise '[Error] Please specify id_column parameter to use drop duplicates functionality.'
        gdf[geom_colum+'__area_size'] = gdf[geom_colum].area
        gdf = gdf.sort_values(geom_colum+'__area_size', ascending=False)
        gdf = gdf.drop_duplicates(subset=[id_column], keep='first')
        gdf = gdf.drop([geom_colum+'__area_size'], axis=1)
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

def add_buffer_column(gdf, buffer_radius, geom_column = 'geometry', buffer_shape = 'round', proj2 = None, replace = False, return_new_column = False):

    radius_marker = re.sub('000$','k',str(int(buffer_radius)))
    
    new_column = geom_column if replace else geom_column + '_buffer_'+radius_marker
    
    proj2 == '+proj=cea' if proj2 == 'cea' else proj2
    
    if proj2:
        gdf[new_column] = gdf[geom_column].to_crs(proj2).buffer(buffer_radius, cap_style=buffer_shape).to_crs(gdf.crs)
    else:
        gdf[new_column] = gdf[geom_column].buffer(buffer_radius, cap_style=buffer_shape)
        
    if return_new_column:
        return gdf, new_column
    return gdf
  
def left_spatial_join(gdf1, gdf2):
    joined = gpd.sjoin(gdf1, gdf2, how='left')
    if 'index_right' in joined.columns:
        joined = joined.drop(['index_right'], axis=1)
    return joined
    
def add_intersection_count_column(gdf, uuid_column, buffer_column, feature_layer, new_column, feature_geom_column = 'geometry'):
    temp = gpd.sjoin(gdf.set_geometry(buffer_column), feature_layer[[feature_geom_column]], how='left').dropna(subset=['index_right'])
    uuid_to_intersection_count_mapping = temp.groupby(uuid_column)['index_right'].count().to_dict()
    gdf[new_column] = gdf[uuid_column].map(uuid_to_intersection_count_mapping).fillna(0).apply(int)
    return gdf

def add_bounds_column(gdf, geom_column = 'geometry'):

    bounds_columns = gdf[geom_column].bounds
    bounds_column_names = [geom_column+'__'+col for col in bounds_columns.columns]
    bounds_columns.columns = bounds_column_names
    if len(set(bounds_column_names).intersection(set(gdf.columns)))>0:
        raise '[Error] Bounds columns already exist. '
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


def get_raster_value_distribution(gdf, geom_column, uuid_column, geotiff_filepath_column, geotiff_filepath_list, code_to_label_mapping, label_marker, normalize):
    
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
    
    data_df = pd.DataFrame(data_rows, columns=[id_column,'label_count_mapping'])
    label_count_df = pd.json_normalize(data_df['label_count_mapping'])
    label_count_df.columns = label_marker+'__'+label_count_df.columns
    if normalize:
        label_count_df = label_count_df.div(label_count_df.sum(axis=1), axis=0)
    label_count_df[uuid_column] = data_df[uuid_column]
    gdf = gdf.merge(label_count_df, on=uuid_column, how='left')
    return gdf

def drop_bounds_columns(gdf, geom_column='geometry'):
    gdf = gdf.drop([geom_column+'__'+col for col in ['minx', 'maxx', 'miny', 'maxy']], axis=1)
    return gdf

def add_intersection_count_column(gdf, uuid_column, buffer_column, new_layer, new_column, geom_column = 'geometry', new_layer_geom_column = 'geometry', default_right_index = 'index_right'):
    temp = gpd.sjoin(gdf.set_geometry(buffer_column), new_layer[[new_layer_geom_column]], how='left').dropna(subset=[default_right_index])
    uuid_to_intersection_count_mapping = temp.groupby(uuid_column)[default_right_index].count().to_dict()
    gdf[new_column] = gdf[uuid_column].map(uuid_to_intersection_count_mapping).fillna(0).apply(int)
    return gdf



def dist_to_arclength(chord_length, R):
    """
    https://en.wikipedia.org/wiki/Great-circle_distance
    Convert Euclidean chord length to great circle arc length
    """
    central_angle = 2*np.arcsin(chord_length/(2.0*R)) 
    arclength = R*central_angle
    return arclength

def distance_to_nearest_neighbor_using_kdtree(data, R = 6367):
    "Based on https://stackoverflow.com/q/43020919/190597"

    phi = np.deg2rad(data['Latitude'])
    theta = np.deg2rad(data['Longitude'])
    data['x'] = R * np.cos(phi) * np.cos(theta)
    data['y'] = R * np.cos(phi) * np.sin(theta)
    data['z'] = R * np.sin(phi)
    tree = scipy_spatial.KDTree(data[['x', 'y', 'z']])
    distance, index = tree.query(data[['x', 'y','z']], k=2)
    return dist_to_arclength(distance[:, 1], R)

def add_distance_to_nearest_neighbor_column(gdf, geom_centroid_column, new_column, rounding = 0):
    lonlat_df = pd.concat([gdf[geom_centroid_column].x, gdf[geom_centroid_column].y], axis=1)
    lonlat_df.columns = ['Longitude','Latitude']
    distances = distance_to_nearest_neighbor_using_kdtree(lonlat_df[['Latitude','Longitude']])*1000
    distances = distances.astype(int) if rounding == 0 else np.round(distances,rounding)    
    gdf[new_column] = distances
    return gdf


def get_groupby_stats_df(data, groupby_column, stats_map):
    stats_df = data.groupby(groupby_column).agg(stats_map)
    stats_df.columns = ['__'.join(col).strip() for col in stats_df.columns.values]
    return stats_df

def get_most_correlated_feature(data, target, features):
    return data[[target]+features].corr().iloc[1:,0].apply(abs).nlargest(1).index.tolist()[0]


def within_value_to_range_value(gdf, buffer_radius_markers):
    for i in range(len(buffer_radius_markers)-1,0,-1):
        outer_buffer = buffer_radius_markers[i]
        inner_buffer = buffer_radius_markers[i-1]
        for col in [col for col in gdf.columns if col.endswith(outer_buffer)]:
            feature_base = col.replace(outer_buffer,'')
            gdf[feature_base+outer_buffer.split('_')[-1]+'-'+inner_buffer.split('_')[-1]] = gdf[col] - gdf[feature_base+inner_buffer]
    for col in [col for col in gdf.columns if col.endswith(buffer_radius_markers[0])]:
        gdf = gdf.rename(columns = {col:col.replace('within_','0-')})    
    gdf = gdf.drop([col for col in gdf.columns if '_within_' in col], axis=1)
    return gdf


### Classification

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

def plot_precision_recall_curve(model, testX, testy, model_name = 'Model', return_thresholds = False):
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(testX)
    lr_precision, lr_recall, thresholds = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print()
    # plot the precision-recall curves
    no_skill = len(testy[testy==1]) / len(testy)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label=model_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plot title
    plt.title(model_name + ' f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    if return_thresholds:
        return thresholds


