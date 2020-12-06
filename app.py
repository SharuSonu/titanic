#!/usr/bin/env python

from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import joblib
import flask

model = joblib.load('titanic.pkl')
cols=['pclass','age','fare','adult_male','alone','sex_male','alive_yes']

app=Flask(__name__,template_folder='templates')

@app.route("/", methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template("main.html")

    if flask.request.method == 'POST':
        input_data=[]
    
        for col in cols:
            input_data.append(float(request.form[col]))

        pred=model.predict([input_data])

        if pred == 1:
                return 'Survived'
        else:
            return 'Didnt Survive'

if __name__=='__main__':
    app.run()

