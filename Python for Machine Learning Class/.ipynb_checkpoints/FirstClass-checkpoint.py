import pandas as pd

dataset = pd.read_csv('paris_airbnb.csv')

X = dataset.iloc[0].values
