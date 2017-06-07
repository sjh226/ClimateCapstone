import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

def gaus_p(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    gpr = GaussianProcessRegressor()
    gpr.fit(X_train, y_train)
    print('Score of GPR: {}'.format(gpr.score(X_test, y_test)))
    return gpr


if __name__ == '__main__':
    df = pd.read_pickle('40yr_df.pkl')
    df = df.iloc[::100, :]
    y = df.pop('date').values
    X = df.values
    gpr = gaus_p(X, y)
