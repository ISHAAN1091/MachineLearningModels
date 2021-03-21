import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Loading the data
df = pd.read_csv('./data/mushrooms.csv')
print(df.head())
print(df.shape)

# Encoding the categorical data into numerical data
le = LabelEncoder()
# .apply method is used to apply a method on each column or row of the dataframe
ds = df.apply(le.fit_transform)
print(ds.head())
print(ds.shape)
# Converting the ds dataframe into a numpy array now
data = ds.values
print(data.shape)
print(type(data))
print(data[:5, :])
# Also breaking the data into X and Y arrays
data_X = data[:, 1:]
data_Y = data[:, 0]
print(data_X.shape, data_Y.shape)

# Break the data into train and test
x_train, y_train, x_test, y_test = train_test_split(
    data_X, data_Y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Checking how many unique classes of mushroom we have
np.unique(y_train)

# De-encoding data back from numerical data to categorical data
# For this purpose you can use inverse_transform method . Read more about it in documentation of scikit-learn


# Building our classifier
