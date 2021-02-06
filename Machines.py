import warnings
from math import sqrt

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error


def scaler(x):
    scl = MinMaxScaler()
    scl.fit(x)
    return scl.transform(x)


def submission(m, x, y):

    model = m().fit(x, y)
    teste = pd.read_csv('test')

    Id = teste['Id']
    teste = teste.iloc[:, 1:]

    Y_hat = model.predict(teste)
    Y_hat = pd.Series(Y_hat.reshape(len(Y_hat,)))

    submission = pd.concat([Id, Y_hat], axis=1)
    submission.columns=['Id', 'SalePrice']
    submission.to_csv('submission.csv', sep=',', index=False)


def run_class(m, x, y, linear, **kwargs):
    x_tr, x_t, y_tr, y_t = train_test_split(x, y, test_size=.3)
    print(m.__module__)
    if m.__module__ == 'sklearn.ensemble.forest':
        out = m(class_weight='balanced').fit(x_tr, np.ravel(y_tr))
    elif m.__module__ == 'sklearn.svm.classes':
        out = m(class_weight='balanced', gamma=kwargs['gamma'] if kwargs else 10).fit(x_tr, np.ravel(y_tr))
    else:
        out = m().fit(x_tr, np.ravel(y_tr))
    y_hat = out.predict(x_t)
    if not linear:
        accuracy = metrics.accuracy_score(y_t, y_hat)
        print('Accuracy: {:.04f}'.format(accuracy))
    else:
        print('Coeficiente de determinação: {:.2f}'.format(out.score(x, y)))
        print('Root mean squared error: {:,.2f}'.format(sqrt(mean_squared_error(y_t, y_hat))))


def main(x, y):
    class_models = [SVC,
                    RandomForestClassifier]
    linear_models = [LinearRegression,
                     DecisionTreeRegressor]
    for each in class_models:
        run_class(each, x, y, False)
    for each in linear_models:
        run_class(each, x, y, True)


def run_single(m, x, y, a, b, v):
    d = dict()
    for i in range(a, b):
        print('Testing with {}={}'.format(v, i))
        d[v]=i
        run_class(m, x, y, False, **d)


if __name__ == '__main__':
    X = pd.read_csv('X')
    Y = pd.read_csv('Y')
    X = scaler(X)
    main(X, Y)

    # run_single(SVC, X, Y, 2, 10, 'gamma')
    # run_class(DecisionTreeRegressor, X, Y, False)
    # run_class(LinearRegression, X, Y, False)
    submission(LinearRegression, X, Y)
