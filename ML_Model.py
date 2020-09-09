import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import  SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from preprocessing import prepare_Data

"""
    score take the original data (features , predicted val ground truth )
    and apply prediction internal then compare 
    y ground truth with the y predicted in some statistical manner 
"""

def Linear_Regression(Data , poly=False , degree=2 , bias=True):
    X = Data.iloc[: , 1:]
    Y = Data["Average User Rating"]
    if poly:
        poly = PolynomialFeatures(degree=degree, include_bias=bias)
        X = poly.fit_transform(X)
    reg = linear_model.LinearRegression()
    reg.fit(X , Y)  # i guess it do the trainig
    Y_pred = reg.predict(X)
    print("mean square error is {}".format(mean_squared_error(Y , Y_pred)))
    print("acuracy is : {}".format(reg.score(X , Y)))
    return reg

def SV_regression(Data , poly=False , degree=2 , bias=True):
    """
    :param Data:  pandas data frame
    :return: SVR model
    """
    X = Data.iloc[: , 1:]
    Y = Data.iloc[: , 0]

    if poly:
        poly = PolynomialFeatures(degree=degree, include_bias=bias)
        X = poly.fit_transform(X)

    sv_reg = SVR( C=1.0 , epsilon=0.2)
    sv_reg.fit(X , Y)
    Y_pred = sv_reg.predict(X)
    print("mean square error is {}".format(mean_squared_error(Y , Y_pred)))
    print("accuracy is {}".format(sv_reg.score(X , Y)))
    return sv_reg

def predict_regression(input , reg , prep = False):
    """
    :param input:  supposed to be pandas frame
    :param reg:
    :return:
    """
    if not prep:
        input = prepare_Data(input)

    Avg_user_rate = reg.predict([input])
    print("average user rate is " , Avg_user_rate)
    return Avg_user_rate

def conv_to_poly(X , degree=2 , bias=True):
    poly = PolynomialFeatures(degree=degree, include_bias=bias)
    X = poly.fit_transform(X)
    return X;