import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def gaus_p(X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_trs = scaler.transform(X_train)
    X_tes = scaler.transform(X_test)
    X_trs = X_train
    X_tes = X_test
    # combine square exponential with periodic to create locally periodic
    # kernel
    rbf_ls = 70
    ess_ls = 10
    # 1 year = 0.0318719999999999 before scaled
    per = .03
    kernel = RBF(length_scale=rbf_ls) * ExpSineSquared(length_scale=ess_ls, periodicity=per)
    gpr = GaussianProcessRegressor(alpha=.5, \
                                   kernel=kernel, \
                                   normalize_y=True)
    gpr.fit(X_trs, y_train)

    y_pred, std = gpr.predict(X_tes, return_std=True)

    print('Using parameters...\nRBF ls: {}, Sine ls: {}, period: {}'\
          .format(rbf_ls, ess_ls, per))
    print('Score of GPR: {}'.format(gpr.score(X_tes, y_test)))

    plot_pred(gpr, 'Gaussian Process', 'train_predict', X_trs, y_train, X_tes)

    return gpr, y_pred

def plot_pred(model, model_name, fig_name, X_train, y_train, X_test):
    plt.close()
    y_pred = model.predict(X_train)
    x = np.linspace(X_train.min(), X_train.max(), 50).reshape(-1, 1)
    y = model.predict(x)

    # plot data
    plt.scatter(X_train, y_train, c='k', label='Test Data', s=5, alpha=0.7)
    # plt.scatter(X_train, y_pred, c='r', label='Predictions', s=5, alpha=0.7)
    plt.plot(x, y, c='r', label='Prediction', linewidth=2)
    plt.title('{} Climate Predictions'.format(model_name))
    plt.xlabel('Time')
    plt.ylabel('Temperature (F)')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    # plot prediction as regression
    y_pred, std = model.predict(X_test, return_std=True)
    # plt.plot(X_test, y_pred, 'b-', label='Prediction')
    # plt.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([y_fut - 1.9600 * std,
    #                         (y_fut + 1.9600 * std)[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='95% confidence interval')


        # plt.plot(future, y_fut)
        # plt.fill_between(future[:, 0], y_fut - std, y_fut + std,\
        #                  alpha=0.5, color='k')
        # plt.xlim(X_test.min(), future.max())
    plt.legend()
    plt.savefig('{}.png'.format(fig_name))


if __name__ == '__main__':
    df = pd.read_pickle('40yr_df.pkl')
    df = df.iloc[::100, :]
    df = df[df['date'] > '2005']
    train_df = df.loc[df['date'] < '2014']
    test_df = df.loc[df['date'] >= '2014']
    for data in [train_df, test_df]:
        data['date'] = pd.to_numeric(data['date'], errors='coerce')/1000000000000000000

    X_train = train_df['date'].values.reshape(-1, 1)
    X_test = test_df['date'].values.reshape(-1, 1)
    y_train = train_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)
    y_test = test_df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)

    gpr, y_pred = gaus_p(X_train, X_test, y_train, y_test)

    # all_X = df['date'].values.reshape(-1, 1)
    # all_y = df['hourly_dry_bulb_temp_f'].values.reshape(-1, 1)
    #
    # predicts = pd.DataFrame(\
    #             np.array(['2017-06-01', '2018-01-01',\
    #                      '2018-06-01', '2019-01-01',\
    #                      '2019-06-01', '2020-01-01',\
    #                      '2020-06-01', '2021-01-01',\
    #                      '2021-06-01', '2022-01-01',\
    #                      '2022-06-01', '2023-01-01',], dtype='datetime64'))
    # predicts = (pd.to_numeric(predicts[0].values)/1000000000000000000)\
    #             .reshape(-1, 1)
    #
    #
    # y = df.pop('date').values
    # X = df.values
    # gpr = gaus_p(X, y)
