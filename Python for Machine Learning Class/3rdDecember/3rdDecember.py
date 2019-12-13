import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree


dataset = pd.read_csv('wines-quality.csv', sep=",")
dataset.head(n=5)


X = dataset.drop('quality', axis=1)

y = dataset.quality
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




dataset = dataset.replace({'red':1, 'white': 2}) 
X = dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [12])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


X