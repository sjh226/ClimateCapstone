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
        y, std = pred_one(model, x)
        if len(x) == 10:
            years = list(range(2000, (int(x[:4]) + 1)))
            std_years = list(range(2017, (int(x[:4]) + 1)))
            dates = ['{}{}'.format(year, x[4:]) for year in years]
            std_dates = ['{}{}'.format(year, x[4:]) for year in std_years]
            y_preds = np.array([pred_one(model, str(date))[0] for date in dates])
            std_preds = np.array([pred_one(model, str(date))[0] for date in std_dates])
            stds = np.array([pred_one(model, str(date))[1][0] for date in std_dates])
            print(std_preds, stds)
            plt.close()
            plt.plot(years, y_preds, c='r')
            plt.fill(np.concatenate([std_years, std_years[::-1]]),\
                     np.concatenate([std_preds - 1.96 * stds, (std_preds + 1.96 * stds)[::-1]]),\
                     alpha=.5, fc='gray', ec='None', label='95% confidence interval')
            plt.title('Prediction for {} Since 2000'.format(x[5:]))
            plt.xlabel('Date')
            plt.ticklabel_format(style='plain', axis='x')
            plt.ylabel('Temperature (C)')
            if int(x[:4]) > 2017:
                plt.legend()
            plt.tight_layout()
            plt.savefig('static/img/pred_{}.png'.format(x))
            return render_template('predict.html', y_pred=y, Date=x)
        else:
            return render_template('false_predict.html')
    except:
        return render_template('false_predict.html')

if __name__ == '__main__':
    with open('model/gpr_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    app.run(host='0.0.0.0', port=8080, debug=True)
