import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from build_data_set import *
from Visualization import *
from ML_Model import *
import FileOperations as fop

FILE_NAME = "predicting_mobile_game_success_train_set.csv"


def nan_statistics(Data):
    total_missing = Data.isnull().values.sum()
    print("total number of rows is : {}".format(Data.shape[0]))
    print("total number of missing entries is : {}".format(total_missing))

    print("\n statistics for each feature and number of examples that miss that feature \n")
    print(Data.isnull().sum())
    return


# parameters
poly = False
degree = 2
bias = True

# prepare the data

Data = fop.ReadData(FILE_NAME)

# filter the data corresponding to our decisions
Data = prepare_Data(Data)
# Data['Developer Genre'] = Data['Developer'] * Data['Primary Genre']  #for combining two features
# normalize the data
Data = full_normalization(Data)

# save the data set
fop.save_data_set(Data, "game_dataset_normalized")

Data = shuffle(Data)
# SPLIT train_test_set  is doing this in random
X_train, X_valid, Y_train, Y_valid = split_data(Data)
train_data = pd.concat([Y_train, X_train], axis=1)
valid_data = pd.concat([Y_valid, X_valid], axis=1)
if poly == True:
    X_valid = conv_to_poly(X_valid, degree, bias)
"""
# linear regression model
model = Linear_Regression(train_data, poly, degree, bias)

# support vector regressiom
# model = SV_regression(train_data ,poly ,  degree , bias)

# save the model
# fop.save_model(model, 'svr_deg3')
"""
# load the model
model = fop.load_model('linear')

# get MSE and accuracy
#print(X_valid.shape)
Y_pred = model.predict(X_valid)

print("mean square error is {}".format(mean_squared_error(Y_valid, Y_pred)))
print("accuracy is ", model.score(X_valid, Y_valid))
