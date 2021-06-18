#==============================================================================
## Group plume locations
#==============================================================================
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
import pdb
import sys
from datetime import datetime as dt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.speedups
import geopandas as gpd
from pandarallel import pandarallel
import rasterio
#==============================================================================


def remove_bb_plumes(plume_df,bb_file_path,odiac_data,bb_distance = 10):

    fire_df = pd.read_csv(bb_file_path)

    # Remove low condifence fires
    fire_df = fire_df[fire_df.confidence != 'l']
    if fire_df.acq_date[0] != '2020-06-01':
        # Only include vegetation fires
        fire_df = fire_df[fire_df.type == 0]

    # Reduce down to only data needed:
    fire_df = fire_df[['acq_date','latitude','longitude']]

    bb_removed = plume_df.copy(deep = True)
    grouped_dates = plume_df.groupby('Date')

    max_fire_distance = bb_distance # km between plume and fire location

    bb_df = pd.DataFrame(columns = ['Date', 'Centre_Latitude', 'Centre_Longitude',
                            'NO2_Column','Vertex_Lats', 'Vertex_Lons', 'Filename'])

    for plume_date, result in grouped_dates:

        ff_points = zip(result.Centre_Longitude,result.Centre_Latitude)
        ff_pix = odiac_data.sample(ff_points)

        ff_pix = [bool(f) for f in ff_pix]

        fire_dates = fire_df[fire_df.acq_date == plume_date]
        
        fire_dates[['Lat_Rad','Lon_Rad']] = np.radians(fire_df.loc[:,['latitude','longitude']])
        result[['Lat_Rad','Lon_Rad']] = np.radians(result.loc[:,['Centre_Latitude','Centre_Longitude']])
        dist = neighbors.DistanceMetric.get_metric('haversine')
        
        dist_matrix = dist.pairwise(result[['Lat_Rad','Lon_Rad']],fire_dates[['Lat_Rad','Lon_Rad']]) * 6371
        df_dist = pd.DataFrame(dist_matrix)
        plume_nums = np.asarray(df_dist.index)
       
        for n,row in enumerate(plume_nums):
            nearby_plumes = np.where(df_dist.loc[row].lt(max_fire_distance) == True)[0]
            
            if ff_pix[n]: # If the pixel is also found in the ODIAC pixel then skip
                 
                continue
            if len(nearby_plumes) :
                plume_index = result.index[row]
                bb_removed.drop(plume_index, inplace = True)
                bb_df = bb_df.append(plume_df.loc[plume_index], ignore_index= True)

    return bb_removed, bb_df

def list_from_str(input_str):
    removed_brackets = input_str.split('[')[1].split(']')[0]
    str_list = removed_brackets.split()
    to_float_list = [float(x) for x in str_list]
    return to_float_list

def remove_bb_plumes_new(plume_df,bb_file_path):

    fire_df = pd.read_csv(bb_file_path, index_col = 'acq_date',parse_dates = True)
    # Remove low condifence fires
    fire_df = fire_df[fire_df.confidence != 'l']
    # Only include vegetation fires
    fire_df = fire_df[fire_df.type == 0]

    # Reduce down to only data needed:
    fire_gdf  = gpd.GeoDataFrame(geometry = gpd.points_from_xy(
        fire_df.longitude,fire_df.latitude
    ), index = fire_df.index)

    plume_df.Vertex_Lats = plume_df.Vertex_Lats.apply(list_from_str)
    plume_df.Vertex_Lons = plume_df.Vertex_Lons.apply(list_from_str)
    
    fire_pixel_arr = []

    grouped_dates = plume_df.groupby('Date')
    for plume_date, result in grouped_dates:
        print(plume_date)
       
        plume_polys = []
        for ind, plume in result.iterrows():
            lat_verts = plume.Vertex_Lats
            lon_verts = plume.Vertex_Lons
            
            plume_pixel = [None]*4

            for pc in range(4):
                plume_pixel[pc] = (lon_verts[pc],lat_verts[pc])
            plume_polys.append(Polygon(plume_pixel))

        plume_gdf = gpd.GeoDataFrame(geometry = plume_polys)

        day_fire = fire_gdf[fire_gdf.index == plume_date]
        day_fire.reset_index(inplace = True)
        fire_geom = day_fire.geometry 
        pixel_geom = plume_gdf.geometry 

        within_matrix = pixel_geom.parallel_apply(lambda x: fire_geom.within(x)).values.astype(int)
        fire_pixel = within_matrix.sum(axis = 1)
        in_fire = fire_pixel != 0
        fire_pixel_arr = np.concatenate((fire_pixel_arr,in_fire))
    return fire_pixel_arr


if shapely.speedups.available:
    shapely.speedups.enable()

pandarallel.initialize()

if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('INCORRECT NUMBER OF INPUT VALUES! YEAR AND MONTH NEEDED')

year = int(sys.argv[1])
month = int(sys.argv[2])

# Path and Name of files containing plume locations (output from plume_finder.py)
fname = 'plume_locations_{}_0.75qa_nonan.csv'.format(str(year)+str(month).zfill(2))
print('Opening: {}'.format(fname))
plume_df = pd.read_csv(fname)
plume_df.drop('Unnamed: 0',axis =1 , inplace = True)

viirs_file = 'VIIRS_BB_{}.csv'.format(str(year)+str(month).zfill(2))

# fire_pixel_arr = remove_bb_plumes_new(plume_df,viirs_file)
# breakpoint()
# BB_removed = plume_df[fire_pixel_arr == 0]
# BB_removed.reset_index(drop = True, inplace = True)
# OnlyBB = plume_df[fire_pixel_arr == 1]
# OnlyBB.reset_index(drop = True, inplace = True) # Reset index from 0 - end

# outfile = '/geos/d21/dfinch/plume_ml_data/plume_locations_{}_0.75qa_BB_removed_updated.csv'.format(str(year)+str(month).zfill(2))
# print('Creating file: {}'.format(outfile))
# BB_removed.to_csv(outfile)

# BBoutfile = '/geos/d21/dfinch/plume_ml_data/plume_locations_{}_0.75qa_OnlyBB_updated.csv'.format(str(year)+str(month).zfill(2))
# print('Creating file: {}'.format(BBoutfile))
# OnlyBB.to_csv(BBoutfile)

plume_df,bb_df = remove_bb_plumes(plume_df,viirs_file,ODIAC_data,bb_distance = 15)
plume_df.reset_index(drop = True, inplace = True)


outfile = 'plume_locations_{}_BB_removed.csv'.format(str(year)+str(month).zfill(2))
print('Creating file: {}'.format(outfile))
plume_df.to_csv(outfile)

BBoutfile = 'plume_locations_{}_OnlyBB.csv'.format(str(year)+str(month).zfill(2))
print('Creating file: {}'.format(BBoutfile))
bb_df.to_csv(BBoutfile)
