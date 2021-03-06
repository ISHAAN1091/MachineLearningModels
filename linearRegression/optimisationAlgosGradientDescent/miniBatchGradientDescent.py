import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, make_regression

# Get or Create a dataset
X, Y = make_regression(n_samples=10000, n_features=20,)

# Normalisation
u = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X-u)/std

# Add a column of x0=1 for vectorization
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))
print(X.shape, Y.shape)

# Creating the hypothesis function


def hypothesis(X, theta):
    return np.dot(X, theta)


# Creating the cost function
def costFunction(X, Y, theta):
    J = 0.0
    h_theta = hypothesis(X, theta)
    J = np.sum((h_theta-Y)**2)
    m = X.shape[0]
    J /= 2*m
    return J


# Creating the function to find gradient
def gradient(X, Y, theta):
    h_theta = hypothesis(X, theta)
    grad = np.dot(X.T, (h_theta-Y))
    m = X.shape[0]
    grad /= m
    return grad


# Batch Gradient Descent
def gradientDescent(X, Y, max_iters=100, alpha=0.01):
    n = X.shape[1]
    theta = np.zeros((n,))
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
        theta = theta - alpha*grad
    return theta, cost_function_values, theta_values


data = np.hstack((X, Y.reshape(-1, 1)))


# Mini Batch Gradient Descent
def miniBatchGradientDescent(X, Y, batch_size=200, max_iters=100, alpha=0.01):
    n = X.shape[1]
    theta = np.zeros((n,))
    cost_function_values = []
    theta_values = []
    for i in range(max_iters):
        # Mini Batch Gradient Descent
        # Shuffling the data in each iteration
        np.random.shuffle(data)
        m = X.shape[0]
        total_batch_size = m//batch_size
        for i in range(total_batch_size):
            batch_data = data[batch_size*i:batch_size*(i+1), :]
            # Computing the gradient
            grad = gradient(batch_data[:, :-1], batch_data[:, -1], theta)
            # Updating theta to a better value to minimise cost function
            theta = theta - alpha*grad
            # Also calculating and storing cost function for each iteration in cost_function_values
            cost_function_value = costFunction(
                batch_data[:, :-1], batch_data[:, -1], theta)
            cost_function_values.append(cost_function_value)
            # Also storing theta value for each iteration in theta_values
            theta_values.append((x for x in theta))
    return theta, cost_function_values, theta_values


# Getting results using Batch Gradient Descent
theta, cost_function_values, theta_values = gradientDescent(X, Y)
plt.style.use('seaborn')
plt.plot(cost_function_values)
print(theta)
# Getting predictions for the test data using batch gradient descent
predictions = []
for i in range(X.shape[0]):
    pred = hypothesis(X[i], theta)
    predictions.append(pred)
predictions = np.array(predictions)
print(predictions.shape)


# Finding R2 score for batch gradient descent
def r2Score(Y, h_theta_):
    # Here to find the sums instead of using a loop np.sum is recommended as it is faster
    numerator = np.sum((h_theta_-Y)**2)
    denominator = np.sum((Y-Y.mean())**2)
    score = 1 - numerator/denominator
    score *= 100
    return score


# Finding coefficient of determination for our model
print('Coefficient of Determination: ')
r2_score = r2Score(Y, predictions)
print(r2_score)

# Getting results using Mini Batch Gradient Descent
theta, cost_function_values, theta_values = miniBatchGradientDescent(X, Y)
plt.plot(cost_function_values)
plt.show()
print(theta)
# Getting predictions for the test data using mini batch gradient descent
predictions = []
for i in range(X.shape[0]):
    pred = hypothesis(X[i], theta)
    predictions.append(pred)
predictions = np.array(predictions)
print(predictions.shape)


# Finding R2 score for mini batch gradient descent
def r2Score(Y, h_theta_):
    # Here to find the sums instead of using a loop np.sum is recommended as it is faster
    numerator = np.sum((h_theta_-Y)**2)
    denominator = np.sum((Y-Y.mean())**2)
    score = 1 - numerator/denominator
    score *= 100
    return score


# Finding coefficient of determination for our model
print('Coefficient of Determination: ')
r2_score = r2Score(Y, predictions)
print(r2_score)
