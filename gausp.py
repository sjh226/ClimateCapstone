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

    # # example plots for exp lin kernel
    # # Parameters for the expanded exponential kernel
    # theta = 2.0, 50.0, 0.0, 1.0
    # # Some sample training points.
    # xvals = np.random.rand(10) * 2 - 1
    # # Construct the Gram matrix
    # C = covariance(exponential_linear_kernel, xvals, xvals, theta)
    # # Sample from the multivariate normal
    # y, sigma = np.transpose(predictions)
    # yvals = np.random.multivariate_normal(np.zeros(len(xvals)), C)
    # x_pred = np.linspace(-1, 1, 1000)
    # predictions = [predict(i, xvals, exponential_linear_kernel, theta, C, yvals)
    #                for i in x_pred]
    df = pd.read_pickle('40yr_df.pkl')
    y = df.pop('hourly_dry_bulb_temp_f').values
    X = df['date'].values
    kernel = GPy.kern.Matern32(1, variance=10, lengthscale=1.2)
    model = GPy.models.GPRegression(X=X[:,None], Y=y[:,None],kernel=kernel)
    model.optimize()
    model.plot()
