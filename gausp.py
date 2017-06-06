import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')
import GPy

random.seed(13131313)

# credit to https://github.com/fonnesbeck/Bios8366

covariance = lambda kernel, x, y, params: \
    np.array([[kernel(xi, yi, params) for xi in x] for yi in y])

exponential_kernel = lambda x, y, params: params[0] * \
    np.exp( -0.5 * params[1] * np.sum((x - y)**2) )

# sigma = covariance(exponential_kernel, x, x, theta)
theta = 2.0, 50.0, 0.0, 1.0

def conditional(x_new, x, y, fcov=exponential_kernel, params=theta):
    B = covariance(fcov, x_new, x, params)
    C = covariance(fcov, x, x, params)
    A = covariance(fcov, x_new, x_new, params)
    mu = np.linalg.inv(C).dot(B).T.dot(y)
    sigma = A - np.linalg.inv(C).dot(B).T.dot(B)
    return(mu.squeeze(), sigma.squeeze())

def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new

def exponential_linear_kernel(x, y, params):
    return exponential_kernel(x, y, params[:2]) + params[2] + params[3] * np.dot(x, y)

def plot_with_error(xvals, yvals, x_pred, y, sigma):
    plt.close()
    plt.errorbar(x_pred, y, yerr=sigma, capsize=0)
    plt.plot(xvals, yvals, "ro")


if __name__ == '__main__':
    # # example plots for exp kernel
    # x = [1.]
    # y = [np.random.normal(scale=sigma0)]
    # sigma1 = covariance(exponential_kernel, x, x, theta)
    # x_pred = np.linspace(-3, 3, 1000)
    # predictions = [predict(i, x, exponential_kernel, theta, sigma1, y) \
    #                for i in x_pred]
    # y, sigma = np.transpose(predictions)

    plt.close()
    df = pd.read_pickle('40yr_df.pkl')
    df = df[df['date'] > '2010-01-01'].iloc[::100, :]
    y = df.pop('hourly_dry_bulb_temp_f').values
    X = df['date'].values
    # kernel = GPy.kern.Matern32(1, variance=2000, lengthscale=1.2)
    # kernel = GPy.kern.Linear()
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X=X.reshape(X.shape[0], 1),\
                                    Y=y.reshape(y.shape[0], 1),kernel=kernel)
    # # attempting to plot this on my own...
    # X_test = np.arange('2010', '2016', dtype='datetime64[D]')
    # X_test = X_test.reshape(X_test.shape[0], 1)
    # posteriorTestY = model.posterior_samples_f(X_test, full_cov=True, size=3)
    # simY, simMse = model.predict(X_test)
    #
    # plt.plot(X_test, posteriorTestY)
    # plt.plot(X, y.reshape(y.shape[0], 1), 'ok', markersize=10)
    # plt.plot(X_test, simY - 3 * simMse ** 0.5, '--g')
    # plt.plot(X_test, simY + 3 * simMse ** 0.5, '--g')

    # GPy method to plot
    model.optimize()
    model.plot()
    
    plt.savefig('gausp.png')
