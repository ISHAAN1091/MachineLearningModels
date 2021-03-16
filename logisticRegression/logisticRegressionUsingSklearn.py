import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data Preparation
# Creating the dataset using numpy
mean_01 = np.array([1, 0.5])
cov_01 = np.array([[1, 0.1], [0.1, 1.2]])
mean_02 = np.array([4, 5])
cov_02 = np.array([[1.2, 0.1], [0.1, 1.3]])
dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)
print(dist_01.shape, dist_02.shape)

# Visualizing our data
plt.style.use('seaborn')
plt.scatter(dist_01[:, 0], dist_01[:, 1], color='red', label='Class0')
plt.scatter(dist_02[:, 0], dist_02[:, 1], color='blue', label='Class1')
plt.xlabel('Feature X1')
plt.xlabel('Feature X2')
plt.legend()
plt.show()

# Storing dist_01 and dist_02 in a single matrix with y as 3 column and shuffling it
data = np.zeros((1000, 3))
data[:500, :2] = dist_01
data[500:, :2] = dist_02
data[500:, -1] = 1
np.random.shuffle(data)
print(data[:10])

# Data splitting
# Splitting the data into training and testing sample
split = int(0.8*data.shape[0])
X_train = data[:split, :-1]
Y_train = data[:split, -1]
X_test = data[split:, :-1]
Y_test = data[split:, -1]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Plotting our training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='Accent')
plt.show()

# Normalising the data
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
X_train = (X_train-x_mean)/x_std
X_test = (X_test-x_mean)/x_std
# Here we also normalised test data as otherwise it wouldn't have fit according to our normalised
# training data. Also we normalised test data using mean and std of our training data as we want shift
# it in proportion to training data.

# Plotting our normalised training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='Accent')
plt.show()

# Using sklearn to implement logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)
theta_0 = model.intercept_
theta_s = model.coef_
print(theta_0, theta_s)

# Getting the predictions using sklearn library
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
print(pred_train[:5])
print(pred_test[:5])

# Checking accuracy using sklearn
print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))
