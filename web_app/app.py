from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '../')
from skgaus import pred_one

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def predict():
    x = str(request.form['Date'])
    y = pred_one(model, x)
    return render_template('predict.html', y_pred=y, Date=x)

if __name__ == '__main__':
    with open('model/gpr_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    app.run(host='0.0.0.0', port=8080, debug=True)
