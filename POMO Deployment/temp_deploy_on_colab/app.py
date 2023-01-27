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


import sys, os, glob, re, time, random, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet_v2 import preprocess_input
from keras.models import load_model
# from keras.preprocessing import image
import keras.utils as image

import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from pyngrok import ngrok
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
# from flask_ngrok import run_with_ngrok

import json
import plotly
import plotly.express as px

from disease_info import *

from fbprophet import Prophet
# import pandas._libs.arrays

import warnings
warnings.filterwarnings('ignore')

# Define a flask app
app = Flask(__name__)

# start ngrok when app is run
# run_with_ngrok(app)

port_no = 5000
ngrok.set_auth_token("2KdKVvWPZia5YkGnclMcSmJGTjF_7XRGo19n8sau4r29JNaPL")
public_url =  ngrok.connect(port_no).public_url


# Model saved with Keras model.save()
MODEL_PATH = 'model_R50.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('\nModel loaded. Start serving...')
print('\nModel loaded. Check http://127.0.0.1:5000/')

def predict_ensemble_voting_classifier(X_test, path : str):
    
    # selecting top 3 models
    if os.path.exists(path):   

        models_df = pd.read_csv(path)
        models_list = list(models_df.sort_values(by = ['Accuracy'], ascending = False)[0:3]['Filename'].values)

        # get model prediction results
        output_list = []
        for model_path in models_list:
            model = pickle.load(open(os.path.join(MODELS_FILE_PATH, model_path), 'rb')) 
            pred = model.predict(X_test)
            output_list.append(pred)
        
        # get predicted values (from votting)
        and_of_pred = (output_list[0] & output_list[1] & output_list[2])
        or_of_pred = (output_list[0] | output_list[1] | output_list[2])
        y_pred_new = np.nan_to_num(and_of_pred // or_of_pred)
        
        return y_pred_new

    else:
        raise Exception(f"[ {path} ] : file not found.")

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
        management_list = management_dict[result_index]
        treatment_list = treatment_dict[result_index]
        print("**"*20)
        time.sleep(5)

        # df = pd.DataFrame({"Class": ["0","1","2","3","4","5"],"Probability": preds[0]})
        df = pd.DataFrame({
            "Class": ["Anthracnose", "Bacterial Blight", "Blight Borer", "Healthy", "Rot", "Fusarium Wilt"],
            "Probability": preds[0]
            })
        fig = px.bar(df, x="Class", y="Probability", color="Class")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # heat = os.path.join('static/heat.jpeg')
        # gCam = os.path.join('static/gCam.jpeg')
        time.sleep(0.3)
        # return render_template('index.html', result=result, file_path=file_path[7:],heat=heat[7:],gCam=gCam[7:],test2=preClass,graphJSON=graphJSON)
        return render_template('index.html', result=result, file_path=file_path[7:], heat=heat, gCam=gCam[8:], test2=preClass, graphJSON=graphJSON, management=management_list, treatment=treatment_list)
    return None

@app.route('/WeatherBasedPomegranateDiseasePrediction', methods=['GET', 'POST'])
def WeatherBasedPomegranateDiseasePrediction():
    return render_template('WeatherBasedPomegranateDiseasePrediction.html')

@app.route('/WeatherBasedDiseasePredict', methods=['GET', 'POST'])
def WeatherBasedDiseasePredict():
    if request.method == 'POST':
        TempC = float(request.form.get('TempC'))
        Humidity = float(request.form.get('Humidity'))
        WindSpeed = float(request.form.get('WindSpeed'))
        Pressure = float(request.form.get('Pressure'))
        Precipitation = float(request.form.get('Precipitation'))
        WeatherDesc = float(request.form.get('WeatherDesc'))
        SunshineHours = float(request.form.get('SunshineHours'))
        SoilMoisure = float(request.form.get('SoilMoisure'))

        print('--- > Input are : ', TempC, Humidity, WindSpeed, Pressure, 
              Precipitation, WeatherDesc, SunshineHours, SoilMoisure)
        input_test = np.array([[TempC, Humidity, WindSpeed, Pressure, 
                                Precipitation, WeatherDesc, SunshineHours, 
                                SoilMoisure]])
        output = predict_ensemble_voting_classifier(input_test, 
                                                    MULTI_MODEL_FILE_PATH)
        print('--- > Output are : ', output)

        # 25    50  5   1014    0.000000    4   9.8 45 == 10000
        # 26    43  9   1012    0.000000    5   9.7 42 == 00000
        # 25    58  5   1013    0.066667    3   9.8 50 == 11000
        # 23    91  26  999 1.733333    1   3.8 70  == 00110 != 00111 
        # 26    89  5   1004    0.766667    1   6.2 65 == 10101
        # 25    78  12  1001    2.200000    1   9.1 64 == 11000 != 11101
        # 23    44  5   1013    0.000000    5   9.8 45 == 00000
        # 24    90  7   1006    0.066667    3   5.4 55 == 00001
        # 24    91  6   1004    0.100000    3   3.8 65  == 00101

        # getting predicated disease by ensemble voting classifier
        pred_disease = [type_of_disease[ind] for ind, val in enumerate(output[0]) if val == 1]
        if len(pred_disease) == 0: pred_disease = ['The weather is suitable for the healthy growth of pomegranate. ']

        pred_disease_management = []
        for ind, val in enumerate(output[0]): 
            if val == 1:
                # pred_disease_management.extend(disease_management_list[ind])
                pred_disease_management.append(disease_management_list[ind])
        if len(pred_disease_management) == 0: pred_disease_management = [['The healthy growth of pomegranate. ']]

        pred_disease_treatment = []
        for ind, val in enumerate(output[0]): 
            if val == 1:
                # pred_disease_treatment.extend(disease_treatment_list[ind])
                pred_disease_treatment.append(disease_treatment_list[ind])
        if len(pred_disease_treatment) == 0: pred_disease_treatment = [['The healthy growth of pomegranate. ']]

    return render_template('WeatherBasedPomegranateDiseasePrediction.html', disease=pred_disease, disease_management=pred_disease_management, disease_treatment=pred_disease_treatment, result=[pred_disease, pred_disease_management, pred_disease_treatment])

@app.route('/DiseaseForecasting')
def DiseaseForecasting():
    return render_template('DiseaseForecasting.html')

@app.route('/next_forecast', methods=['GET', 'POST'])
def next_forecast():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        next_days_range = int(request.form.get('next_days'))
        
        file_path = os.path.join('static', secure_filename(f.filename))
        # Save the file to ./uploads
        f.save(file_path)

        df = pd.read_csv(file_path)
        # print(f"date : {df['observation'].iloc[-1]}")
        from_date = datetime.datetime.strptime(df['observation'].iloc[-1], '%Y-%m-%d').date() # df['observation'].iloc[-1]
        # print(f"from_date : {from_date}")
        next_days = [(from_date + datetime.timedelta(days=day)) for day in range(1, next_days_range+1)]
        # print(f"next_days : {next_days}")
        next_df = pd.DataFrame(next_days, columns=['ds'])

        models_list = ["tempC_avg_0C__pro.pkl", "Relative_humidity_avg__pro.pkl", 
                       "windspeedKmph_avg_Km_h__pro.pkl", "pressureMB_avg_pro.pkl",  
                       "precipMM_avg_mm__pro.pkl", "weatherDesc_pro.pkl", 
                       "Sunshine_Hours_pro.pkl", "__soil_moisure_pro.pkl"]

        model_path_list = [os.path.join('static', 'Forecast_Models', m) for m in models_list]
        # print(f"model_path_list : {model_path_list}")

        col_yhat_out = []
        for m_path in model_path_list:
            model = pickle.load(open(m_path, 'rb')) 
            forecast = model.predict(next_df)
            col_yhat_out.append(list(forecast['yhat']))
        
        data = [next_days] + col_yhat_out
        df3 = pd.DataFrame(data).T
        col_list = ['observation','tempC_avg(0C)', 'Relative humidity_avg(%)', 
                    'windspeedKmph_avg(Km/h)', 'pressureMB_avg', 'precipMM_avg(mm)', 
                    'weatherDesc', 'Sunshine Hours', '%_soil_moisure']
        #rename all column names
        df3.columns = col_list 
        # convert columns to int64 dtype 
        df3 = df3.astype({"tempC_avg(0C)": int, "Relative humidity_avg(%)": int, 
                          "windspeedKmph_avg(Km/h)": int, "pressureMB_avg": int, 
                          "weatherDesc": int, "%_soil_moisure": int})
        df3['windspeedKmph_avg(Km/h)'] = df3['windspeedKmph_avg(Km/h)'].abs()
        # print(df3)

        df3.to_csv('data.csv', index=False)

        df1 = df3.iloc[:, 1:]

        output_list = []
        for ind in range(len(df1)):
            input_test = np.array([list(df1.iloc[ind])])
            output = predict_ensemble_voting_classifier(input_test, 
                                                        MULTI_MODEL_FILE_PATH)
            print(f"Next Day_{ind+1} : output : {output}")
            output_list.append(output[0])

        df_output_list = pd.DataFrame(output_list, columns=['Bacterial Blight/Telya', 'Anthracnose', 
                                                 'Fruit Spot/Rot', 'Fusarium Wilt', 
                                                 'Fruit Borer/Blight Blora'])
        df4 = pd.concat([df3, df_output_list], axis=1)
        # json_data = df4.to_dict() # df4.to_json(orient='values')
        # print(json_data)

        df4.to_csv("output_result.csv", index=False)
        
        Bacterial_dates = df4["observation"][df4['Bacterial Blight/Telya'] == 1].tolist()
        Anthracnose_dates = df4["observation"][df4['Anthracnose'] == 1].tolist()
        FruitSpot_dates = df4["observation"][df4['Fruit Spot/Rot'] == 1].tolist()
        Fusarium_dates = df4["observation"][df4['Fusarium Wilt'] == 1].tolist()
        FruitBorer_dates = df4["observation"][df4['Fruit Borer/Blight Blora'] == 1].tolist()

        df_cols = ['tempC_avg(0C)', 'Relative humidity_avg(%)', 'windspeedKmph_avg(Km/h)', 
           'pressureMB_avg', 'precipMM_avg(mm)', 'weatherDesc', 'Sunshine Hours', 
           '%_soil_moisure']

        fig0 = px.line(df4, x="observation", y=df_cols[0])
        for bd in Bacterial_dates:
            fig0.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig0.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig0.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig0.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig0.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON0 = json.dumps(fig0, cls=plotly.utils.PlotlyJSONEncoder)

        fig1 = px.line(df4, x="observation", y=df_cols[1])
        for bd in Bacterial_dates:
            fig1.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig1.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig1.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig1.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig1.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        fig2 = px.line(df4, x="observation", y=df_cols[2])
        for bd in Bacterial_dates:
            fig2.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig2.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig2.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig2.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig2.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        fig3 = px.line(df4, x="observation", y=df_cols[3])
        for bd in Bacterial_dates:
            fig3.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig3.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig3.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig3.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig3.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        fig4 = px.line(df4, x="observation", y=df_cols[4])
        for bd in Bacterial_dates:
            fig4.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig4.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig4.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig4.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig4.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

        fig5 = px.line(df4, x="observation", y=df_cols[5])
        for bd in Bacterial_dates:
            fig5.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig5.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig5.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig5.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig5.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

        fig6 = px.line(df4, x="observation", y=df_cols[6])
        for bd in Bacterial_dates:
            fig6.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig6.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig6.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig6.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig6.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

        fig7 = px.line(df4, x="observation", y=df_cols[7])
        for bd in Bacterial_dates:
            fig7.add_vline(x=bd, line_width=2, line_dash="dash", line_color="green")
        for ad in Anthracnose_dates:
            fig7.add_vline(x=ad, line_width=3, line_dash="dot", line_color="blue")
        for fsd in FruitSpot_dates:
            fig7.add_vline(x=fsd, line_width=2, line_dash="dot", line_color="red")
        for fd in Fusarium_dates:
            fig7.add_vline(x=fd, line_width=3, line_dash="dot", line_color="yellow")
        for fbd in FruitBorer_dates:
            fig7.add_vline(x=fbd, line_width=2, line_dash="dash", line_color="magenta")
        graphJSON7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)


        return render_template('DiseaseForecasting.html', flag=True, 
                               next_number_of_days=next_days_range,
                               titles=df4.columns.values,
                               data=df4.values.tolist(),
                               graphJSON0=graphJSON0,
                               graphJSON1=graphJSON1,
                               graphJSON2=graphJSON2,
                               graphJSON3=graphJSON3,
                               graphJSON4=graphJSON4,
                               graphJSON5=graphJSON5,
                               graphJSON6=graphJSON6,
                               graphJSON7=graphJSON7)

    return render_template('DiseaseForecasting.html', flag=False)

@app.route('/plot_me')
def plot_me():
    
    df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
    })
   
    fig = px.bar(df, x='Fruit', y='Amount', color='City', 
      barmode='group')
   
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    #  plot 2
    df1 = px.data.gapminder().query("country in ['Canada', 'Botswana']")
    fig1 = px.line(df1, x="lifeExp", y="gdpPercap", 
                   color="country", text="year")
    fig1.update_traces(textposition="bottom right")
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


    # plot 3
    df2 = px.data.iris()
    fig2 = px.scatter(df2, x="petal_length", y="petal_width")
    fig2.add_vline(x=2.5, line_width=3, line_dash="dash", line_color="green")
    fig2.add_vline(x=3.5, line_width=3, line_dash="dash", line_color="blue")
    fig2.add_vline(x=5, line_width=3, line_dash="dash", line_color="red")
    fig2.add_vline(x=5.5, line_width=3, line_dash="dash", line_color="yellow")
    
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('plot_me.html', 
                           graphJSON=graphJSON, 
                           graphJSON1=graphJSON1,
                           graphJSON2=graphJSON2)



if __name__ == '__main__':
    print(f"To acces the Gloable link please click {public_url}")
    # app.run(debug=True)
    app.run(port=port_no)

