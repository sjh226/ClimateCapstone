import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def gaus_p(X_train, X_test, y_train, y_test=None):
    scaler = preprocessing.StandardScaler()
    # scaler.fit(X_train)
    # X_trs = scaler.transform(X_train)
    # X_tes = scaler.transform(X_test)
    X_trs = X_train
    X_tes = X_test
    # combine square exponential with periodic to create locally periodic
    # kernel
    rbf_1 = 250
    rbf_2 = 70
    ess_ls = 10
    # 1 year = 0.03187 before scaled
    per = .03
    # kernel for general positive trend
    k1 = RBF(length_scale=rbf_1)
    # kernel for seasonal periodicity
    k2 = RBF(length_scale=rbf_2) * ExpSineSquared(length_scale=ess_ls, \
                                                   periodicity=per)
    kernel = k1 + k2
    gpr = GaussianProcessRegressor(alpha=.5, \
                                   kernel=kernel, \
                                   normalize_y=True)
    gpr.fit(X_trs, y_train)

    y_pred, std = gpr.predict(X_tes, return_std=True)

    if y_test != None:
        print('Score of GPR: {}'.format(gpr.score(X_tes, y_test)))

    plot_pred(gpr, 'Gaussian Process', 'pred_test', X_trs, y_train, X_tes)

    return gpr, y_pred

def plot_pred(model, model_name, fig_name, X_train, y_train, X_test):
    plt.close()
    y_pred = model.predict(X_train)
    x = np.linspace(X_train.min(), np.max(X_test), 100).reshape(-1, 1)
    y = model.predict(x)

    # plot data
    plt.scatter(X_train, y_train, c='k', label='Test Data', s=5, alpha=0.7)

    # plot prediction as regression
    plt.plot(x, y, c='r', label='Prediction', linewidth=2)

    # x_pred = np.linspace(X_train.max(), np.max(X_test), 100).reshape(-1, 1)
    # y_pred, std = model.predict(x_pred, return_std=True)
    # plt.plot(x_pred, y_pred, c='r', linewidth=2)
    # # x_pred = np.linspace(X_train.max(), np.max(X_test), 1000).reshape(-1, 1)
    # # y_pred, std = model.predict(x_pred, return_std=True)
    # # plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
    # #          np.concatenate([y_pred - 1.9600 * std,
    # #                         (y_pred + 1.9600 * std)[::-1]]),
    # #          alpha=.3, fc='b', ec='None', label='95% confidence interval')
    #
    # plt.fill_between(x_pred, y_pred - std, y_pred + std,\
    #                  alpha=0.5, color='k')

    plt.title('{} Climate Predictions'.format(model_name))
    plt.xlabel('Time')
    plt.ylabel('Temperature (F)')
    # plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    # plt.legend()
    plt.savefig('images/{}.png'.format(fig_name))


if __name__ == '__main__':
    df = pd.read_pickle('40yr_df.pkl')
    # work with 100th of the data for simplicity
    df = df.iloc[::100, :]
    # start with dates over 2007, predict on dates past 2015
    # this is the most populated, clean data
    df = df[df['date'] > '2007']
    train_df = df.loc[df['date'] < '2015']
    test_df = df.loc[df['date'] >= '2015']
    # convert datetime to seconds since epoch *e19
    for data in [train_df, test_df]:
        data['date'] = pd.to_numeric(data['date'], errors='coerce')/1000000000000000000
    # manual train/test split based on year
    X_train = train_df['date'].values.reshape(-1, 1)
    X_test = test_df['date'].values.reshape(-1, 1)
    y_train = train_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)
    y_test = test_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)

    gpr, y_pred = gaus_p(X_train, X_test, y_train, y_test)

    # df['date_epoch'] = pd.to_numeric(df['date'], errors='coerce')/1000000000000000000
    # all_X = df['date_epoch'].values.reshape(-1, 1)
    # all_y = df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)
    #
    # predicts = pd.DataFrame(\
    #             np.array(['2018-01-01'], dtype='datetime64'))
    # predicts = (pd.to_numeric(predicts[0].values)/1000000000000000000)\
    #             .reshape(-1, 1)
    #
    # gpr, y_pred = gaus_p(all_X, predicts, all_y)

    # Using parameters...
    # RBF ls: 70, Sine ls: 10, period: 0.03
    # On data since 2005
    # Score of GPR: 0.4582533725449096
