import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib 


#Init Flask 
app = Flask(__name__)

#Loading the Model File
model = joblib.load('aqi_pro.pkl')
sc=joblib.load("scaler.pkl")

#To launch home page
@app.route('/')
def home():
    #loading a HTML page
    return render_template('web.html')

#prediction page
@app.route('/y_predict', methods=['POST'])
def y_predict():
    x_col = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    data = [[x for x in request.form.values()]]
    print(data)
    data = pd.DataFrame(data, columns=x_col)
    data=sc.transform(data)
    prediction = model.predict(data)
    print(prediction)
    result = "AQI: ",prediction[0]
    return render_template('web.html',prediction_text=result)
     
if __name__ == "__main__":
    app.run(debug=True)
