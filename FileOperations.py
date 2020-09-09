import pickle
import pandas as pd

def ReadData(file_name):
   data_set = pd.read_csv(file_name)
   data_set.drop(data_set.columns[data_set.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
   # drop Unnamed problem
   return data_set

def save_model(model , filename):
    with open(filename , 'wb') as f:
        pickle.dump(model , f)

    return


def load_model(filename):
    model = None
    with open(filename , 'rb') as f:
        model = pickle.load(f)
    return model

def save_data_set(Data , name):
    Data.to_csv(name+".csv")