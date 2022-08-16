# Run Flask on Colab

# !pip install flask-ngrok 
# !pip install gevent
# !pip install pandasql


# from flask_ngrok import run_with_ngrok
# from flask import Flask

# # # Running the flask app
# app = Flask(__name__)

# # # start ngrok when app is run
# run_with_ngrok(app)

# @app.route("/")
# def index():
#   return "<h1>Home Page!! </h1>"

# @app.route("/about")
# def about():
#   return "<h1>About Page!! </h1>"

# @app.route("/us")
# def us():
#   return "<h1>Us Page!! </h1>"

# app.run()



# # To unzip main files
# from zipfile import ZipFile
# file_name = "/content/drive/MyDrive/Colab Notebooks/Project_Files.zip"
# print(file_name)
# with ZipFile(file_name, 'r') as zip:
#   zip.extractall()
#   print('done')


# !pip install Flask-Caching


from __future__ import division, print_function
# coding=utf-8
import sys, os, glob, re, time, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet_v2 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_ngrok import run_with_ngrok

import json
import plotly
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# Define a flask app
app = Flask(__name__)

# start ngrok when app is run
# run_with_ngrok(app)

# Model saved with Keras model.save()
MODEL_PATH = 'model_R50.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('\nModel loaded. Start serving...')
print('\nModel loaded. Check http://127.0.0.1:5000/')

management_dict = {
    0 : ["Wide row spacing. ", "Selection of disease free seedlings for fresh planting. ", "Pruning affected branches, fruits regularly and burning. ", "Bahar should be done in Haste or Ambe Bahar. ", "Give minimum four month rest after harvesting the fruits. "],
    1 : ["Clean cultivation and aintenance of health and vigour of the tree should be followe. ", "The fruits if screened with polythene or paper bags may escape infestation. ", "Removal and destruction of all the affected fruits."],
    2 : ["The plant is healthy. "],
    3 : ["Affected fruits should be collected and destroyed. ", "Select Haste or Ambe bahar. ", "Wider plant spacing, yearly pruning of trees. "],
    4 : ["Do not allow water to stagnate, try to create drainage facility. ", "Do not irrigate for 2-3 days after drenching. "]
} 


# def save_and_display_gradcam(img_path, heatmap, cam_path="static//gCam.jpeg", alpha=0.99):
def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.99):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def model_predict(img_path, model):
    img_size = (224, 224)
    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Print what the top predicted class is
    preds = model.predict(img_array)
    # print("Predicted:", preds)

    preClass = np.argmax(preds)
    # print('Class : ', preClass)

    last_conv_layer_name = "conv5_block3_3_conv"  #"block14_sepconv2_act"
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    # plt.savefig('static//heat.jpeg', dpi=300)
    heat_name = 'Heat_Img_'+str(random.randint(1, 1000))+'.jpeg'
    plt.savefig('static//'+heat_name, dpi=300)
    # plt.show()
    cam_path = 'static//gCam_Img_'+str(random.randint(1, 1000))+'.jpeg'
    save_and_display_gradcam(img_path, heatmap, cam_path)

    return preds, preClass, heat_name, cam_path

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        file_path = os.path.join('static', secure_filename(f.filename))
        # Save the file to ./uploads
        f.save(file_path)

        # Make prediction
        preds, preClass, heat, gCam = model_predict(file_path, model)
        result = str(preds[0])
        # print('----> \t\t heat, gCam : ', heat, gCam)
        print("----> \t\t result : ", result)
        preds_result_index = list(preds[0])
        result_index = preds_result_index.index(max(preds_result_index))
        print("----> index value : ", result_index )
        print("**"*20)
        time.sleep(5)

        # df = pd.DataFrame({"Class": ["0","1","2","3","4","5"],"Probability": preds[0]})
        df = pd.DataFrame({
            "Class": ["Anthracnose","Bacterial Blight","Blight Borer","Healthy","Rot","Fusarium Wilt"],
            "Probability": preds[0]
            })
        fig = px.bar(df, x="Class", y="Probability", color="Class")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # heat = os.path.join('static/heat.jpeg')
        # gCam = os.path.join('static/gCam.jpeg')
        time.sleep(0.3)
        # return render_template('index.html', result=result, file_path=file_path[7:],heat=heat[7:],gCam=gCam[7:],test2=preClass,graphJSON=graphJSON)
        return render_template('index.html', result=result, file_path=file_path[7:],heat=heat,gCam=gCam[8:],test2=preClass,graphJSON=graphJSON)
    return None

@app.route('/WeatherBasedPomegranateDiseasePrediction', methods=['GET', 'POST'])
def WeatherBasedPomegranateDiseasePrediction():

    return render_template('WeatherBasedPomegranateDiseasePrediction.html')

if __name__ == '__main__':
    app.run(debug=True)
    # app.run()

