import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

vixdataset = pd.read_csv('vixcurrent.csv')
vixchange = vixdataset.Open

callput = pd.read_csv('spxpc.csv')
callputratio = callput.ratio

feargreed = pd.read_csv('feargreed.csv')
feargreed = feargreed.FearGreed

pricing = pd.read_csv("SPYprices.csv")
listed = []

for i in pricing.index:
    if i < 2198:
        if pricing.iloc[i+3, 5] - pricing.iloc[i, 5] > 0:
            listed.append(10)
        else:
            listed.append(0)

listed.append(10)

frame = {
    'Fear Greed': feargreed,
    'CallPutRatio': callputratio,
    'VixChange': vixchange,
    '3daysfromnow': listed
    }

df = pd.DataFrame(frame)

X = df[['Fear Greed', 'CallPutRatio', 'VixChange']]
Y = df['3daysfromnow']

tx, ex, ty, ey = train_test_split(X, Y, random_state=3)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for i in range(2, 300, 5):
    returned = get_mae(i, tx, ex, ty, ey)
    print("Average loss of model with {} node is {}".format(i, returned))