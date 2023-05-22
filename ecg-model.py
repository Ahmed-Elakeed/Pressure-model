from flask import Flask, request, render_template, url_for, jsonify
import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

model=load_model("best_model(2)(1).h5")
UPLOAD_FOLDER = 'folders/'
app = Flask(__name__)
def preprocessing(recieveecg):
        ecgPath = 'folders/ecg.csv'
        recieveecg.save(ecgPath)
        ecg = pd.read_csv(ecgPath)
        ecg = ecg.to_numpy()
        file_names = ["newww(1).csv"] 
        n_features = 32 
        csv_header = ["f"+str(i) for i in range(n_features+1)] 
        csv_header = ",".join(csv_header)
        data = []
        for file in file_names:
           label = 1
           print("Processing file " + file)
           ecg_signal = np.loadtxt(os.path.join(folderPath, file), delimiter=",")
           for j in range(0, 10 , 10):
             AR, rho, ref = arburg(ecg_signal, n_features)
             features = [k.real for k in AR]
             features.append(rho)
             features.append(label)
             data.append(features)
        c=np.savetxt("test.csv", data, delimiter=",", header=csv_header, comments="")
        return c



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        recieveecg=request.files['ecg']
        d = preprocessing(recieveecg)
        firstprediction = model.predict(d)

    return render_template('index.html', prediction = firstprediction )

@app.route('/homeAPI', methods=['GET','POST'])
def homeAPI():
    try:
        if 'ecg' not in request.files:
            return "Please try again. The file doesn't exist"
        recieveecg=request.files['ecg']
        d = preprocessing(recieveecg)
        firstprediction = model.predict(d)
        return jsonify({'prediction': firstprediction})
    except:
        return jsonify({'Error': 'Error occur'})

if __name__ == '__main__':
    app.run(debug=True)
