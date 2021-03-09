import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression

# Creating dataset
X, Y = make_regression(n_samples=400, n_features=1,
                       n_informative=1, noise=1.8, random_state=11)
# Also converting Y into a matrix
Y = Y.reshape((-1, 1))
print(X.shape, Y.shape)

# Normalising the data
X_ = (X-X.mean())/X.std()

# Visualizing our dataset
plt.figure()
plt.style.use('seaborn')
plt.scatter(X, Y)
plt.show()

# Adding a column of x0=1 in X
X_ = np.hstack((np.ones((X_.shape[0], 1)), X_))
print(X.shape)


# This function is the hypothesis function or the function that would in the end return us the prediction
def predict(X, theta):
    h_theta = np.dot(X, theta)
    return h_theta


# Using normal equation we find the value of theta
def getThetaClosedForm(X, Y):
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return theta


theta = getThetaClosedForm(X_, Y)
print(theta)

# Visualizing our dataset along with the best fit line
plt.figure()
plt.scatter(X, Y)
plt.plot(X, predict(X_, theta), color='red', label='prediction')
plt.legend()
plt.show()
