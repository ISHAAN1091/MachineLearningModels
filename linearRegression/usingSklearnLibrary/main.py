import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Creating a regression dataset
# n_samples - used to tell the number of samples required in the dataset
# n_features - used to tell the number of features required in the dataset
# n_informative - used to tell the number of features out of all the features
# on which Y depends that is this feature is noninformative
# noise - The standard deviation of the gaussian noise applied to the output.
# random_state - used to define a key for random dataset creation in order to reproduce the same random
# datset again when code is run again rather creating an entirely new dataset next time
X, Y = make_regression(n_samples=1000, n_features=2,
                       n_informative=2, noise=10, random_state=1)
print(X.shape, Y.shape)
print(type(X), type(Y))

# Visualizing our dataset
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], Y)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 1], Y)
plt.show()

# Plotting in 3D
fig = plt.figure(figsize=(10, 7))
axes = plt.axes(projection='3d')
axes.scatter3D(X[:, 0], X[:, 1], Y, color='green')
plt.title('3D scatter plot')
plt.show()

# Implementing linear regression using sklearn library
# defining the model
model = LinearRegression()
# fitting our data
model.fit(X, Y)
# Accessing the parameters of the best line
# To get from theta1 to thetaN we use .coef_
print(model.coef_)
# To get theta0 we use intercept_ as theta0 is stored separately
print(model.intercept_)
# Getting the predictions
# For example here we try to find the predictions for X[0] and X[1]
print(model.predict([X[0], X[1]]))
# Printing Y[0] and Y[1] to see if our predictions are any good
print(Y[0], Y[1])
# Finding R2 score of our model it gives a value between 0 and 1
print(model.score(X, Y)*100)
