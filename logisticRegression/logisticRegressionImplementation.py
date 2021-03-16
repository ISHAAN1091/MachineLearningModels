import numpy as np
import matplotlib.pyplot as plt

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


# Implementing Logistic Regression
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def hypothesis(X, theta):
    """
    Here X is mXn+1 matrix
    theta is n+1X1 
    """
    return sigmoid(np.dot(X, theta))


def costFunction(X, Y, theta):
    h_theta = hypothesis(X, theta)
    err = (Y*np.log(h_theta)+(1-Y)*np.log(1-h_theta))
    err = -1*err.mean()
    return err


def gradient(X, Y, theta):
    h_theta = hypothesis(X, theta)
    grad = np.dot(X.T, (Y-h_theta))
    m = X.shape[0]
    grad /= m
    return grad


def gradient_descent(X, Y, alpha=0.5, max_iters=100):
    n = X.shape[1]
    theta = np.zeros((n, 1))
    cost_function_values = []
    theta_values = []
    for i in range(max_iters):
        # Computing the gradient
        grad = gradient(X, Y, theta)
        # Also calculating and storing cost function for each iteration in cost_function_values
        cost_function_value = costFunction(X, Y, theta)
        cost_function_values.append(cost_function_value)
        # Also storing theta value for each iteration in theta_values
        theta_values.append((x for x in theta))
        # Updating theta to a better value to minimise cost function
        theta = theta + alpha*grad
    return theta, cost_function_values, theta_values


# Plotting the sigmoid function
a = np.linspace(-10, 10, 20)
plt.scatter(a, sigmoid(a))
plt.show()


# Also we need to add a column of ones that is a column for x0 in our matrix X_train
ones = np.ones((X_train.shape[0], 1))
X_new_train = np.hstack((ones, X_train))
print(X_new_train.shape)

# Also reshaping Y_train
Y_train = Y_train.reshape((-1, 1))

# Fitting the parameters
theta, cost_function_values, theta_values = gradient_descent(
    X_new_train, Y_train)
plt.plot(cost_function_values)
plt.show()

# Visualizing the Decision Surface
plt.scatter(X_train[:, 0], X_train[:, 1],
            c=Y_train.reshape((-1,)), cmap='Accent')
x1 = np.arange(-3, 4)
x2 = -(theta[0]+theta[1]*x1)/theta[2]
plt.plot(x1, x2, color='red')
plt.show()


# Getting the predictions and finding the accuracy
# Adding x0 in X_test
ones = np.ones((X_test.shape[0], 1))
X_new_test = np.hstack((ones, X_test))
print(X_new_test.shape)


def predict(X, theta):
    h = hypothesis(X, theta)
    output = np.zeros(h.shape)
    output[h >= 0.5] = 1
    output = output.astype('int')
    return output


# Finding predictions on training data and test data
pred_train = predict(X_new_train, theta)
pred_test = predict(X_new_test, theta)


# Defining function to find the accuracy
def accuracy(actual, preds):
    actual = actual.astype('int')
    actual = actual.reshape((-1, 1))
    acc = np.sum(actual == preds)/actual.shape[0]
    return acc*100


acc_train = accuracy(Y_train, pred_train)
acc_test = accuracy(Y_test, pred_test)
print(acc_train)
print(acc_test)
