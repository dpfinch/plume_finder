#==============================================================================
## Train image recognition model to detect plumes
#==============================================================================
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import h5py
import random
import numpy as np
import pandas as pd
import pdb
#==============================================================================

def load_dataset(data_file_path, label_file_path, train_perc = 50, normalise = True):
    # Load in the images/data from which to train and test the model
    # Set the train_size to determine how much data to train the data with
    # The remaining data will be used to test the model
    
    all_imgs = h5py.File(data_file_path,'r')
    img_tiles = all_imgs['NO2 Plumes'][:]
    img_tiles = np.nan_to_num(img_tiles, nan = 0.0)
    all_imgs.close()

    if normalise:
    # Need to normalise the data for the model to be efficient
        for img in range(len(img_tiles)):
            img_tiles[img] = img_tiles[img]/np.max(img_tiles[img]) 
    
    # Read in the image labels  
    img_labels = pd.read_csv(label_file_path)

    # img_labels = img_labels[img_labels['Use Image']]
    # img_labels = img_labels.drop(['Use Image'], axis = 1)

    # img_labels.reset_index(inplace = True)

   # Randomly shuffle the order of the images.
    img_index = list(range(len(img_labels)))
    random.shuffle(img_index)

    # Turn percentage train perc into actual number of images
    train_size = int(len(img_labels) * (train_perc/100))

    print('Training on {} images'.format(train_size))

    train_index = img_index[:train_size]
    test_index = img_index[train_size:]

    train_data = img_tiles[train_index]
    #  model will expect 4 dims (N,x,y,z) so reshape to add 4th dim
    train_data = train_data.reshape(-1,28,28,1)
    train_labels = img_labels['Has Plume'].loc[train_index].astype(int).values

    test_data = img_tiles[test_index]
    # Reshape test data too
    test_data = test_data.reshape(-1,28,28,1)
    test_labels = img_labels['Has Plume'].loc[test_index].astype(int).values

    return train_data,train_labels,test_data,test_labels    

def load_alternative_data(data_file_path, label_file_path, train_perc = 50, normalise = True):
    # Load in the images/data from which to train and test the model
    # Set the train_size to determine how much data to train the data with
    # The remaining data will be used to test the model
    
    all_imgs = h5py.File(data_file_path,'r')
    img_tiles = all_imgs['NO2 Plumes'][:]
    img_tiles = np.nan_to_num(img_tiles, nan = 0.0)
    all_imgs.close()

    if normalise:
    # Need to normalise the data for the model to be efficient
        for img in range(len(img_tiles)):
            img_tiles[img] = img_tiles[img]/np.max(img_tiles[img]) 
    
    # Read in the image labels  
    img_labels = pd.read_csv(label_file_path)

    # Turn percentage train perc into actual number of images
    train_size = int(len(img_labels) * (train_perc/100))

    print('Training on {} images'.format(train_size))

    train_data = img_tiles[:train_size]
    #  model will expect 4 dims (N,x,y,z) so reshape to add 4th dim
    train_data = train_data.reshape(-1,28,28,1)
    train_labels = img_labels['Label'].astype(int).values[:train_size]

    test_data = img_tiles[train_size:]
    # Reshape test data too
    test_data = test_data.reshape(-1,28,28,1)
    test_labels = img_labels['Label'].astype(int).values[train_size:]

    return train_data,train_labels,test_data,test_labels

def create_model(img_shape_x, img_shape_y):
    # Create the structure for the neural network model
     
    # Initialise a model
    model = Sequential()

    # Add layers 2 convolution
    # filters = Start with 32 filters
    # kernal size = height and width of conv window (maybe how small the smallest image can be?)
    # padding = "same" -> padding the image to account for any plumes at edge of image
    model.add(Conv2D(32, (2, 2), padding='same', input_shape=(img_shape_x,img_shape_y,1), activation="relu"))
    model.add(Conv2D(64, (2, 2), activation="relu"))
    # Max pooling - just extract important information from the image in 2x2 pixels
    # reduces the information in the image by a factor of 2*2 (4) for ease of processing
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout removes a given fraction of the data to force the model to try harder to learn patterns
    # This ensures model doesn't start to rely too heavily on same characteristics 
    model.add(Dropout(0.5))

    # Repeat the processes again, increasing the number of filters by a factor of 2
    model.add(Conv2D(128, (2, 2), padding='same', activation="relu"))
    model.add(Conv2D(256, (2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.75))

    # Repeat the processes again, increasing the number of filters by a factor of 2
    # model.add(Conv2D(512, (2, 2), padding='same', activation="relu"))
    # model.add(Conv2D(512, (2, 2), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output into single array
    model.add(Flatten())
    # Add a neural network layer with 512 nodes
    model.add(Dense(512, activation="relu"))
    # Increase the dropout fraction to really test the model
    model.add(Dropout(0.75))
    # Add layer with 2 possible outputs (yes/no)
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam', # adam = adaptive moment estimation
        metrics=['accuracy']
    )

    return model

# Set some GPU memory growth protocols to prevent running out of memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Create the model. Images are 28x28
model = create_model(28,28)

model.summary()

# Set the file path containing the test data. Currently needs to be a hdf file
# containing and array of the data.

data_file_path = '' # path to datafile here
label_file_path = '' # path to label file here

# train_data,train_labels, test_data, test_labels = load_dataset(
#     data_file_path,label_file_path,train_perc = 60) # 1565 images in this data, choose % of data to train (rest to test)

train_data,train_labels, test_data, test_labels = load_alternative_data(
    data_file_path,label_file_path,train_perc = 80) # 3795 images in this data, choose % of data to train (rest to test)

# Train the model
model.fit(
    train_data,
    train_labels,
    batch_size=32,
    epochs=50, # Iterate through the model x times
    validation_data=(test_data, test_labels),
    shuffle=True # Make sure to train the model in a random order 
)

loss, acc = model.evaluate(test_data,test_labels)
print(" Model accuracy: {:5.2f}%".format(100*acc))

# Save neural network structure
model_structure = model.to_json()
f = Path("plume_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("plume_model_weights.h5")

## ============================================================================
## END OF PROGAM
## ============================================================================

