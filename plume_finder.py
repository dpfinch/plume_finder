#==============================================================================
## Train image recognition model to detect plumes
#==============================================================================
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from pathlib import Path
import h5py
from netCDF4 import Dataset
import random
import numpy as np
from numpy import unravel_index
import pandas as pd
from glob import glob
from datetime import datetime as dt
from matplotlib import pyplot as plt
import sys
import pdb
#==============================================================================
if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('INCORRECT NUMBER OF INPUT VALUES! YEAR AND MONTH NEEDED')

year = int(sys.argv[1])
month = int(sys.argv[2])

# Set some GPU memory growth protocols to prevent running out of memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Load the json file that contains the model's structure
f = Path("plume_model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("plume_model_weights.h5")

# Set the file path containing the TROPOMI data.
s5p_file_path = ''
# years = [2018,2019]
#month = 11
plume_df = pd.DataFrame({'Date':[],'Centre_Latitude':[], 'Centre_Longitude':[], 'NO2_Column':[],
       'Vertex_Lats':[], 'Vertex_Lons':[], 'Filename':[]})
# for year in years:
all_files = glob('{}{}/{}/*/*.nc'.format(s5p_file_path, str(year), str(month).zfill(2),
    recursive = True))
#print('{}{}/{}/*/*.nc'.format(s5p_file_path, str(year), str(month)))

all_files.sort()
print('Found {} files for {} {}'.format(len(all_files), year, month))
lat_indices = []
lon_indices = []
timestamp = []
i = 1

all_percs = []

for fname in all_files:
    # print('Processing file: {}'.format(fname.split('/')[-1]))
    date = dt.strptime(fname.split('____')[-1].split('_')[0][:8],'%Y%m%d')
    dataset = Dataset(fname)
    products = dataset.groups['PRODUCT']
    no2 = products.variables['nitrogendioxide_tropospheric_column'][0]
    # no2 = products.variables['sulfurdioxide_total_vertical_column'][0]
    lat = products.variables['latitude'][0]
    lon = products.variables['longitude'][0]
    qa = products.variables['qa_value'][0]

    lat_bounds = products['SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0]
    lon_bounds = products['SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0]

    dataset.close()

    qa_limit = 0.75 # 0.5 for cloudy scenes. 0.75 for cloud free

    no2_data = no2.filled(np.nan)
    no2_data[qa < qa_limit] = np.nan

    x_res = 28 # pixels 
    y_res = 28

    plumes = []
    all_tiles = []
    lat_tiles = []
    lon_tiles = []
    lon_bound_tiles = []
    lat_bound_tiles = []
    not_normed_data = []

    data_subset_len = 0

    for y in range(0,3127,y_res): # Trim off the ends of the swatch since these will be messed up by the poles anyway
        for x in range(0,450,x_res):
            data_subset = no2_data[y:y+y_res,x:x+x_res]
            data_subset_len += 1
            try:
                np.nanmax(lat[y:y+y_res,x:x+x_res])
            except ValueError:
                continue
            if np.nanmax(lat[y:y+y_res,x:x+x_res]) > 75 or np.nanmin(lat[y:y+y_res,x:x+x_res]) < -66:
                continue
##                try:
##                    land_mask = globe.is_land(lat[y:y+y_res,x:x+x_res],lon[y:y+y_res,x:x+x_res])
##                except ValueError:
##                    continue
##                if land_mask.sum()<10:
##                    continue
            data_subset[data_subset <= 0] = np.nan
            if np.count_nonzero(~np.isnan(data_subset)) < 300:
                continue

            if data_subset.shape != (28,28):
                continue
            
            not_normed_data.append(data_subset)
            data_subset_fill = np.nan_to_num(data_subset, nan = 0.0)
            data_subset_normed = data_subset_fill/np.max(data_subset_fill)
            all_tiles.append(data_subset_normed)
            lat_tiles.append(lat[y:y+y_res,x:x+x_res])
            lon_tiles.append(lon[y:y+y_res,x:x+x_res])
            lat_bound_tiles.append(lat_bounds[y:y+y_res,x:x+x_res,:])
            lon_bound_tiles.append(lon_bounds[y:y+y_res,x:x+x_res,:])
##                data_reshaped = data_subset_normed[None,:,:,None]

##                prediction = model.predict(data_reshaped)
##                if prediction[0][1] > 0.75:
##                    index = unravel_index(data_subset_normed.argmax(), data_subset_normed.shape)
##                    lat_indices.append(lat[y:y+y_res,x:x+x_res][index])
##                    lon_indices.append(lon[y:y+y_res,x:x+x_res][index])
##                    timestamp.append(date)
                
##                if ypre_list[0][1] > 0.75:
##                    plt.imshow(data_subset)
##                    plt.savefig('/Users/dfinch/Desktop/temp/{}_y.png'.format(str(i).zfill(3)))
##                    plt.close()
##                else:
##                    plt.imshow(data_subset)
##                    plt.savefig('/Users/dfinch/Desktop/temp/{}_n.png'.format(str(i).zfill(3)))
##                    plt.close()
##                plumes.append(ypre_list[0][1])
##                i += 1

    # print('Total images in this file: {}'.format(data_subset_len))
    # print('Total images being used: {}'.format(len(all_tiles)))
    perc_images = (len(all_tiles)/data_subset_len) * 100
    # print('Percentage of images being used: {}%'.format(perc_images))
    all_percs.append(perc_images)

    if len(all_tiles) == 0:
            #  I think this will only happen for file S5P_OFFL_L2__NO2____20190403T112920_20190403T131050_07620_01_010300_20190409T105852.nc
        continue
    tile_array= np.dstack(all_tiles)
    tiles_reshaped = np.rollaxis(tile_array,-1)
##        tile_filled = np.nan_to_num(tiles_reshaped)
##        tiles_normed = tile_filled/np.max(tile_filled)
####        tiles_normed = tile_filled/temp_max
    tiles_data = tiles_reshaped.reshape(-1,28,28,1)
##
    ypre_list = model.predict(tiles_data)
    # ypre_list is a 2D numpy array with
    #  1st column = chance of no plume
    #  2nd column = chance of plume 
    plumes = []
    for x in ypre_list:
        plumes.append(x[1])
    plumes = np.asarray(plumes)
    no2_tiles = np.asarray(tiles_data)
    not_normed_data = np.asarray(not_normed_data)
    lat_tiles = np.asarray(lat_tiles)
    lon_tiles = np.asarray(lon_tiles)
    lat_bound_tiles = np.asarray(lat_bound_tiles)
    lon_bound_tiles = np.asarray(lon_bound_tiles)
##
    lat_indices = []
    lon_indices = []
    timestamp = []
    no2_filename = []
    plume_val = []
    lat_pixel_bounds = []
    lon_pixel_bounds = []
    # Pick where there is more than 0.75 chance there is a plume in the image
    # Arbitrarily chosen
    for p in np.where(plumes>0.9)[0]:
        data_tile = no2_tiles[p,:,:,0]
        # plt.imshow(data_tile)
        # plt.savefig('/geos/d21/dfinch/plume_ml_data/plume_images_found/ML_plume_{}.png'.format(str(i).zfill(5)))
        # plt.close()
        i += 1
        index = unravel_index(data_tile.argmax(), data_tile.shape)
        lat_index = lat_tiles[p][index]
        lat_indices.append(lat_index)
        lon_index = lon_tiles[p][index]
        lon_indices.append(lon_index)
        timestamp.append(date)
        no2_filename.append(fname.split('/')[-1])
        plume_val.append(not_normed_data[p][index])
        lat_pixel_bounds.append(lat_bound_tiles[p][index])
        lon_pixel_bounds.append(lon_bound_tiles[p][index])

##
    df = pd.DataFrame({'Date':timestamp,'Centre_Latitude':lat_indices,'Centre_Longitude':lon_indices,
        'NO2_Column':plume_val,'Vertex_Lats':lat_pixel_bounds,'Vertex_Lons':lon_pixel_bounds, 'Filename': no2_filename})

    plume_df = plume_df.append(df)
print('Created file for {} {} with {} plumes.'.format(year, month, len(plume_df)))
plume_df.to_csv('plume_locations_{}_0.75qa_nonan_high_limit.csv'.format(str(year)+str(month).zfill(2)))

# print('Mean percentage of images used: {}'.format(np.mean(all_percs)))

## ============================================================================
## END OF PROGAM
## ============================================================================

