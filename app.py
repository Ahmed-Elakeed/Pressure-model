from flask import Flask, request, render_template, url_for, jsonify
import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from spectrum import *

model=load_model("dence_lstm512.h5")
model2=load_model("best_model.h5")

UPLOAD_FOLDER = 'folders/'
app = Flask(__name__)
def Ecgpreprocessing(recieveecg):
        ecgPath = 'folders/ecg2.csv'
        recieveecg.save(ecgPath)
        ecg = pd.read_csv(ecgPath)
        ecg = ecg.to_numpy()
        file_names = [ecgPath] 
        n_features = 32 
        csv_header = ["f"+str(i) for i in range(n_features+1)] 
        csv_header = ",".join(csv_header)
        data = []
        for file in file_names:
            label = 1
            print("Processing file " + file)
            ecg_signal = np.loadtxt(ecgPath, delimiter=",")
            for j in range(0, 10 , 10):
                AR, rho, ref = arburg(ecg_signal, n_features)
                features = [k.real for k in AR]
                features.append(rho)
                data.append(features)
        c=np.savetxt("test.csv", data, delimiter=",", header=csv_header, comments="")
        # data.save('folders/data.csv')
        # ecg=pd.read_csv('folders/data.csv')
        return c



@app.route('/ecg')
def Ecgindex():
    return render_template('indexecg.html')

@app.route('/predictecg', methods=['GET','POST'])
def Ecgpredict():
    if request.method == "POST":
        recieveecg=request.files['ecg']
        Ecgpreprocessing(recieveecg)
        ecg=pd.read_csv('test.csv')
        firstprediction = model2.predict(ecg)
        rounded_arr = np.round(firstprediction, 2)
    return render_template('indexecg.html', prediction = rounded_arr )

@app.route('/EcgAPI', methods=['GET','POST'])
def EcgAPI():
    try:
        if 'ecg' not in request.files:
            return "Please try again. The file doesn't exist"
        recieveecg=request.files['ecg']
        Ecgpreprocessing(recieveecg)
        ecg=pd.read_csv('test.csv')
        firstprediction = model2.predict(ecg)
        rounded_arr = np.round(firstprediction[0], 2)
        return jsonify({'prediction': float(rounded_arr)})
    except:
        return jsonify({'Error': 'Error occur'})


def preprocessing(recieveecg, recieveppg):
        ecgPath = 'folders/ecg.csv'
        ppgPath = 'folders/ppg.csv'
        recieveecg.save(ecgPath)
        recieveppg.save(ppgPath)
        ecg = pd.read_csv(ecgPath)
        ppg = pd.read_csv(ppgPath)
        ecg = ecg.to_numpy()
        ppg = ppg.to_numpy()
        # Reshape the arrays to have shape (n, 1)
        print(ecg.shape)
        ecg = ecg[:1000]
        ppg = ppg[:1000]
        # data = np.stack((ppg, ecg), axis=1)
        c = np.stack((ppg, ecg), axis=1)  # stack the arrays along the second axis
        print(c.shape)  # output: (1000, 2, 1)
        d = np.transpose(c, (2, 0, 1))  # transpose the array to get the desired shape
        print(d.shape)  # output: (1, 1000, 2)
        return d

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        recieveecg=request.files['ecg']
        recieveppg = request.files['ppg']
        d = preprocessing(recieveecg, recieveppg)
        firstprediction = model.predict(d)
        prediction=sum((firstprediction[0]*150)+50)/1000

    return render_template('index.html', prediction = prediction )

@app.route('/homeAPI', methods=['GET','POST'])
def homeAPI():
    try:
        if 'ecg' not in request.files and 'ppg' not in request.files:
            return "Please try again. The files doesn't exist"
        recieveecg=request.files['ecg']
        recieveppg = request.files['ppg']
        d = preprocessing(recieveecg, recieveppg)
        firstprediction = model.predict(d)
        prediction=sum((firstprediction[0]*150)+50)/1000
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})

if __name__ == '__main__':
    app.run(debug=True)