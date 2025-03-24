import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

#IMPORT RIDGE REGRESSOR AND STANDARDSCALER PICKLE
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method =='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        #STANDARDIZE THIS INPUT DATA
        new_scaled_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        # PREDICT OUTPU FO RHTIS INPUT
        result = ridge_model.predict(new_scaled_data) ## this will be in the form of a list 

        return render_template('home.html', results = result[0])
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
