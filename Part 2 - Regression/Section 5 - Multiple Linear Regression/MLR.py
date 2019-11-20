import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

#independent variables
X = dataset.iloc[:, :-1].values
#dependent variable(profit)
y = dataset.iloc[:, 4].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Splitting the training set from the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#fitting the multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the test set result
y_pred = regressor.predict(X_test)

#Building an optimal model using Backward Elimination
import statsmodels.formula.api as sm

#appending an array of 1's because recall B0 the constant in the linearRegression formular.
#well it is not included here in the statsmodel like in the regression library
#you have to manually add it
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#process of backward elimination include all the features and remove them one by one
X_opt = X[:, [0, 1, 2, 3, 4, 5]]









































