import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import pandas as pd

Data = pd.read_csv('game_dataset_normalized.csv')
list_heads = Data.head()

data_heads = []
c = 0
for i in list_heads:
    if c == 0:
        c += 1
        continue
    else:
        data_heads.append(i)
        c += 1

df = pd.DataFrame(Data, columns=data_heads)

# print(df)
c1 = 0
for column in df:
    # Select column contents by column name using [] operator
    if c1 == 0:
        Y = column
        c1 += 1

    else:

        ax1 = df.plot.scatter(x=column, y=Y, c='DarkBlue')

        plt.show()

        c1 += 1

