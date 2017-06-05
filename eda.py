import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from d_manage import clean_data, clean_type, fill_nans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import boto3

s3 = boto3.client('s3')


def plot_departure(df, x_col, y_col, title, file_name):
    plt.close()
    y = df[y_col].values
    x = df[x_col].values
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    plt.scatter(x, y, color='black', s=.3, marker='o')
    X_test, y_test, lr = lin_reg(x, y)
    plt.plot(X_test, lr.predict(X_test), color='red', linewidth=1)
    plt.title(title)
    plt.xlabel('Time since Jan 1, 1977')
    plt.ylabel('Temperature (F)')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.savefig('{}.png'.format(file_name))

def lin_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, \
                                                        random_state=13)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print('Score: {}'.format(lr.score(X_test, y_test)))
    return X_test, y_test, lr


if __name__ == '__main__':
    obj = s3.get_object(Bucket='climate_data', Key='40yr.csv')
    df = pd.read_csv(obj['Body'])
    # df = pd.read_csv('965113.csv')
    df, climate_df = clean_data(df)
    df = clean_type(df)

    depart_df = climate_df[climate_df['daily_dept_from_normal_average_temp'].notnull()]\
                  [['date', 'daily_dept_from_normal_average_temp']]
    climate_df = climate_df[climate_df['monthly_dept_from_normal_average_temp'].notnull()]

    for col in depart_df.columns:
        depart_df[col] = pd.to_numeric(depart_df[col], errors='coerce')

    plot_departure(depart_df, 'date', 'daily_dept_from_normal_average_temp',\
                   'Daily Departure from Avg Temp over Time', '40_temp_departure_lr')
    # plot_departure(climate_df, 'date', 'monthly_dept_from_normal_average_temp',\
    #                'Monthly Departure from Avg Temp over Time', 'mo_temp_dep_lr')
