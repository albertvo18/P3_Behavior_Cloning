import argparse
import base64
import json
import cv2

import numpy as np
np.random.seed(42)
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
#tf.python.control_flow_ops = tf
#tf.reset_default_graph()

global last_throttle

last_throttle = 0.0

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]

    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    crop_img = image_array[40:130]
#    crop_img = image_array[30:140]
    height, width = crop_img.shape[:2]
    new_width = int(width/2)
    new_height = int(height/2)
    resized_image = cv2.resize(crop_img,(new_width, new_height),fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    image_array = resized_image
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
#    if abs(steering_angle) > 0.1 and speed > 2:
#        throttle = 0.01
#    else:
#        throttle = 0.15

#    if (abs(speed) < 2):
#    	throttle = 1
#    elif (abs(speed) > 6):
#        throttle = 0.1
#    else:
#        throttle = 0.25
#    TEST = True
    TEST = False
#    TEST2 = True
    TEST2 = False
#    last_throttle = float(throttle)
#    print ("last_throttle= ", last_throttle)
    if TEST:
        last_throttle = float(throttle)
#        if ( last_throttle > 0.9):
#        if ( TEST):
        mythrottle = float(throttle)
        if ( last_throttle != -1.000 and mythrottle == 1.0  ):

#          throttle = -0.35
          throttle = -1.0
#          throttle = -1.00
          last_throttle = -1.0
#          steering_angle = -.4
          print ("last_throttle= ", last_throttle)
          print ("throttle = ", throttle)


        elif ( last_throttle == -1.000 and mythrottle != 1.0  ):

  #          throttle = -0.35
            throttle = .35
  #          throttle = -1.00
            last_throttle = .35
            print ("last_throttle= ", last_throttle)
            print ("throttle = ", throttle)


        elif (abs(speed) < 2 and last_throttle != -1.0):
#        if (abs(speed) < 4):
          throttle = 1
          last_throttle = 1.0
    #    elif (abs(speed) > 3):
#        elif (abs(speed) > 2):
#        elif (abs(speed) > 4):
#        elif (abs(speed) > 10):
        elif (abs(speed) > 20):
            throttle = 0.01
            last_throttle = 0.01
        else:
    #        throttle = 0.25
#            throttle = 0.15

#            throttle = 0.50
#            throttle = 0.35
#            last_throttle = 0.35
            throttle = 0.40
            last_throttle = 0.40
#            throttle = 0.35
#            last_throttle = 0.35
#            throttle = - 0.35
#            throttle = 0.40
    elif TEST2:
        last_throttle = float(throttle)
#        if ( last_throttle > 0.9):
#        if ( TEST):
        mythrottle = float(throttle)
        if (abs(speed) < 2 ):
    #        if (abs(speed) < 4):
          throttle = 1
          last_throttle = 1.0
    #    elif (abs(speed) > 3):
    #        elif (abs(speed) > 2):
    #        elif (abs(speed) > 4):
    #        elif (abs(speed) > 10):
        elif (abs(speed) > 20):
            throttle = 0.01
            last_throttle = 0.01
        else:
    #        throttle = 0.25
    #            throttle = 0.15

    #            throttle = 0.50
    #            throttle = 0.35
    #            last_throttle = 0.35
            throttle = 0.40
            last_throttle = 0.40
    #            throttle = 0.35
    #            last_throttle = 0.35
    #            throttle = - 0.35
    #            throttle = 0.40
    else:
#        throttle = .35
#        throttle = .30
         throttle = .35
#        throttle = .40
#        throttle = .60
#        throttle = .70
#        throttle = .65
#        throttle = .50
    print(steering_angle, throttle, speed)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)


#####################################################
    save_folder = 'run1'
    RECORDING = False
    if RECORDING:
            print("RECORDING THIS RUN ...")
    else:
            print("NOT RECORDING THIS RUN ...")
#####################################################




    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
