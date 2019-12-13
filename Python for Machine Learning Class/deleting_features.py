import pandas as pd 
import numpy as np
pd.options.mode.chained_assignment = None
np.random.seed(1)

paris_listings = pd.read_csv('paris_airbnb.csv')
paris_listings = paris_listings.loc[np.random.permutation(len(paris_listings))]
stripped_commas = paris_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
paris_listings['price'] = stripped_dollars.astype('float')


print(paris_listings.info())
print(paris_listings.head())

drop_columns = ['room_type','city','state','longitude','latitude','zipcode','host_acceptance_rate','host_listings_count','cleaning_fee','security_deposit']


paris_listings = paris_listings.drop(drop_columns, axis = 1)


#standardization formula
#x - is the mean 
#x^ is the mean 

#normalizing columns

normalized_listings = (paris_listings - paris_listings.mean())/paris_listings.std()
normalized_listings['price'] = paris_listings['price']
 

#using sklearn for knn
from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:6000]
test_df = normalized_listings.iloc[6000:]

train_columns = ['accomodations','bedrooms']

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[train_columns], train_df['price'])

predictions = knn.predict(test_df[train_columns])


#use all features for this calculate the msc and rmse etc