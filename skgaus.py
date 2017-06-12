import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import datetime


def gaus_p(X_train, X_test, y_train, y_test=None):
    # scale the full data set based on training data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_trs = scaler.transform(X_train)
    X_tes = scaler.transform(X_test)
    X_trs = X_train
    X_tes = X_test
    # combine square exponential with periodic to track seaonality
    # kernel for general positive trend
    k1 = RBF(length_scale=10)
    # kernel for seasonal periodicity
    # 1 year = 0.03187 before scaled
    k2 = RBF(length_scale=70) * ExpSineSquared(length_scale=10, \
                                                   periodicity=.03)
    kernel = k1 + k2
    # build and fit a GP regressor, setting alpha to deal with noise
    gpr = GaussianProcessRegressor(alpha=2, \
                                   kernel=kernel, \
                                   normalize_y=True)
    gpr.fit(X_trs, y_train)

    y_pred, std = gpr.predict(X_tes, return_std=True)

    # if predicting on known values, show score
    if y_test != None:
        print('Score of GPR: {}'.format(gpr.score(X_tes, y_test)))

    # plot training data and prediction
    plot_pred(gpr, 'Gaussian Process', 'pred!', X_trs, y_train, X_tes)

    return gpr, y_pred

def plot_pred(model, model_name, fig_name, X_train, y_train, X_test):
    plt.close()
    fig, ax = plt.subplots()
    y_pred = model.predict(X_train)
    x = np.linspace(X_train.max(), np.max(X_test), 500).reshape(-1, 1)
    y, std = model.predict(x, return_std=True)

    # plot data
    plt.scatter(X_train, y_train, c='k', label='Train Data', s=5, alpha=0.7)

    # plot prediction as regression
    plt.plot(x, y, c='r', label='Prediction', linewidth=1.75)

    # plot confidence interval
    # x_pred = x[np.where(x>X_train.max())]
    # y_pred = y[np.where(x>X_train.max())]
    # std = std[np.where(x>X_train.max())[0]]
    # plt.fill(np.concatenate([x, x[::-1]]),\
    #          np.concatenate([y - 1.96 * std, (y + 1.96 * std)[::-1]]),\
    #          alpha=.5, fc='gray', ec='None', label='95% confidence interval')

    plt.title('{} Climate Predictions'.format(model_name))
    plt.xlabel('Time')
    plt.ylabel('Temperature (C)')
    labels = np.arange(2007, 2021, 1)
    plt.xticks(np.linspace(x.min(), x.max(), 11), labels, rotation=45)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('images/{}.png'.format(fig_name))


if __name__ == '__main__':
    df = pd.read_pickle('data/40yr_df.pkl')
    # work with 100th of the data for simplicity
    df = df.iloc[::100, :]
    # convert fahrenheit to celcius
    df['hourly_dry_bulb_temp_f'] = (df['hourly_dry_bulb_temp_f'] - 32) * (5/9)
    # start with dates over 2007, predict on dates past 2015
    # this is the most populated, clean data
    df = df[df['date'] > '2007']
    train_df = df.loc[df['date'] < '2015']
    test_df = df.loc[df['date'] >= '2015']
    # convert datetime to seconds since epoch *e19
    for data in [train_df, test_df]:
        data['date'] = pd.to_numeric(data['date'], errors='coerce')\
                                     /1000000000000000000
    # manual train/test split based on year
    X_train = train_df['date'].values.reshape(-1, 1)
    X_test = test_df['date'].values.reshape(-1, 1)
    y_train = train_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)
    y_test = test_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)

    # score and plot on the train/test split
    # gpr, y_pred = gaus_p(X_train, X_test, y_train, y_test)

    # train on all data and predict to the future
    df['date_epoch'] = pd.to_numeric(df['date'], errors='coerce')\
                                     /1000000000000000000
    all_X = df['date_epoch'].values.reshape(-1, 1)
    all_y = df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)

    predicts = pd.DataFrame(\
                np.array(['2020'], dtype='datetime64'))
    predicts = (pd.to_numeric(predicts[0].values)/1000000000000000000)\
                .reshape(-1, 1)

    gpr, y_pred = gaus_p(all_X, predicts, all_y)
