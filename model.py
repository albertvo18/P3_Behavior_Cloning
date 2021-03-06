
# coding: utf-8

import numpy as np
np.random.seed(42)
import tensorflow as tf
#tf.python.control_flow_ops = tf
tf.reset_default_graph()


# imports
import csv
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam, SGD
import json
from keras.callbacks import ModelCheckpoint


# more or less log outputs
debug = True
test_data = False
show_images = False

# read data from csv file
csv_file = 'driving_log.csv'

csv_column_names = ['center','left', 'right', 'steering', 'throttle', 'brake', 'speed']

###
# Reads data from the csv file with the collected data
# Returns two arrays: images and angle (steering angle info)
###
def read_data_from_csv_file(csv_file):
    images, steering = [], []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # print(row['center'], row['steering'])
            steering_value = float(row['steering'])

            images.append(row['center'])
            steering.append(steering_value)

            if (steering_value < 0.35  or steering_value > 0.35):
                # adding left camera images
                images.append(row['left'].strip())
                steering.append(steering_value + 35)
                # adding right camera images
                images.append(row['right'].strip())
                steering.append(steering_value - 35)

    return images, steering

# shift height of the image by a small fraction
HEIGHT_SHIFT_RANGE = 60

def height_shift(img):
    rows, cols, channels = img.shape

    # Translation
#    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    tx = 0
    ty = HEIGHT_SHIFT_RANGE * np.random.uniform() - HEIGHT_SHIFT_RANGE / 2

    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])

    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def brightness_shift(img, bright_value=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

###
# Converts the images and steering data into
# training data and labels np arrays
###
def process_data(images, steering):
    show_images = False
    x,y = [],[]
    # process image
    current_images = len(images)
    if (debug):
        print("Recorded images", current_images)
    for i in range(current_images):
        # take a peek at the images printing out some stats and plotting
        image = mpimg.imread(images[i])
        if (show_images):
            print('Image',images[i],'dimensions:', image.shape,"steering",steering[i])
            plt.imshow(image)
            plt.show()
        img = cv2.imread(images[i]) # reads BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # back to RGB format
        # adjust brightness with random intensity to simulate driving in different lighting conditions
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
#        random_bright = .20+np.random.uniform()
        random_bright = .10+np.random.uniform()
        print(random_bright)
        img[:,:,2] = img[:,:,2]*random_bright
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        new_img = img.astype(np.float32)
        shadow_img = add_random_shadow (new_img)
        images[i] = new_img

        # crop image to get rid of unwanted pixels, like trees, car bonnet, etc
        crop_img = img[40:130]
        height, width = crop_img.shape[:2]
        new_width = int(width/2)
        new_height = int(height/2)
        resized_image = cv2.resize(crop_img,(new_width, new_height),fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        if (show_images):
            print('Resized image dimesions:', resized_image.shape)
            plt.imshow(resized_image)
            plt.show()        
        flip_prob = np.random.random()
        if flip_prob > 0.7:
            # flip the image and reverse the steering angle
            flipped_image = cv2.flip(resized_image, 1)
            flipped_steering = steering[i]*(-1)
            images[i] = flipped_image
            steering[i] = flipped_steering
            if (show_images):
                print('Flipped image dimensions:', flipped_image.shape,'Steering:',flipped_steering)
                plt.imshow(flipped_image)
                plt.show()
        else :
            images[i] = resized_image
        if i > 40:
            show_images = False
    if (debug):
        print("Training images", len(images))
    x = np.array(images)
    #x = np.vstack(images)
    y = np.vstack(steering)
    return x,y

###
# Splits data into training and test sets
###
def split_data(x,y):
    # Split into train and test data (20%)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=3234)
    return x_train, x_val, y_train, y_val

# Shuffle the data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# define model arquitecture
# comma.ai
def get_model(time_len=1):
    channels, height, width = 3, 45, 160  # image format

    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5,
            input_shape=(height, width, channels),
            output_shape=(height, width, channels)))

    #model.add(Convolution2D(32, 3, 3, input_shape=(height, width, channels)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten()) # converts 3D feature maps into 1D vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['mse'])

    return model


# In[10]:

# write generator to feed data into the model
def feed_data_generator (image_train, labels_train, batch_size):
    total = (len(y) // batch_size ) * batch_size
    # number of images in the training set
    num_images = len(image_train)
    # create a random index
    random_index = np.random.choice(num_images, size=batch_size, replace=False)
    while True:
        # select the random images and labels with this random index
        features_batch = image_train[random_index,:]
        labels_batch = labels_train[random_index]
        yield features_batch, labels_batch

# train the model

epochs = 10
#epochs = 100
#epochs = 40
batch_size = 128

def train_model(epochs, batch_size, samples_per_epoch, nb_val_samples):
    model.summary()
    checkpoint = ModelCheckpoint(filepath="./model.h5", verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    history = model.fit_generator(
        feed_data_generator(train_features, train_labels, batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=epochs,
        verbose=1,
        callbacks = callbacks_list,
        validation_data = feed_data_generator(val_features, val_labels, batch_size),
        nb_val_samples = nb_val_samples)

    # list all data in history

    return history

def evaluate_model():
    metrics = model.evaluate(val_features, val_labels, batch_size=32, verbose=1)
    keys = model.metrics_names
    print(keys[0],":",metrics[0])
    print(keys[1],":",metrics[1])





if __name__ == '__main__':
	images_raw, steering_raw = read_data_from_csv_file(csv_file)
    # do the tranformations: resize, trim, brightness, flip
	images_processed,steering_processed = process_data(images_raw, steering_raw)
    # turn into np array
	x = np.array(images_processed)
	y = np.vstack(steering_processed)
    # split data into training and data sets
	train_features, val_features, train_labels, val_labels = split_data(x, y)
	train_features, train_labels = unison_shuffled_copies(train_features, train_labels)
	val_features, val_labels = unison_shuffled_copies(val_features, val_labels)
	samples_per_epoch = ((len(train_labels) // batch_size ) * batch_size)*2
	nb_val_samples = ((len(val_labels) // batch_size ) * batch_size)*2
	model = get_model()
	history = train_model(epochs, batch_size, samples_per_epoch, nb_val_samples)
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
	evaluate_model()
	# save arquitecture as model.json
	with open('./model.json', 'w') as outfile: json.dump(model.to_json(), outfile)
