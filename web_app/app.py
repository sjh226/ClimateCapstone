from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from skgaus import pred_one

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def predict():
    try:
        x = str(request.form['Date'])
        y = pred_one(model, x)
        if len(x) == 10:
            years = list(range(2010, (int(x[:4]) + 1)))
            dates = ['{}{}'.format(year, x[4:]) for year in years]
            y_preds = [pred_one(model, str(date)) for date in dates]
            plt.close()
            plt.plot(years, y_preds)
            plt.title('Prediction for {} Since 2010'.format(x[5:]))
            plt.xlabel('Date')
            plt.ylabel('Temperature (C)')
            plt.tight_layout()
            plt.savefig('static/img/pred_range.png')
            return render_template('predict.html', y_pred=y, Date=x)
        else:
            return render_template('false_predict.html')
    except:
        return render_template('false_predict.html')

if __name__ == '__main__':
    with open('model/gpr_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    app.run(host='0.0.0.0', port=8080, debug=True)
