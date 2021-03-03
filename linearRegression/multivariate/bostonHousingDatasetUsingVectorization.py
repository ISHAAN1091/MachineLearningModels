import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Loading the dataset from sklearn library
boston = load_boston()
X = boston.data
Y = boston.target
print(X.shape, Y.shape)
print(boston.feature_names)
print(boston.DESCR)

# Converting the above dataset into a pandas dataframe
df = pd.DataFrame(X)
df.columns = boston.feature_names
print(df.head())
print(df.describe())
# Note that when we use the .describe() method we observe that the dataset is not normalised i.e. features
# don't have a zero mean and unit standard deviation

# Normalising the data
# That is each feature must have zero mean and unit standard deviation
u = np.mean(X, axis=0)
std = np.std(X, axis=0)
print(u.shape, std.shape)
# Applying normalisation
X = (X-u)/std
# Normalised data
df = pd.DataFrame(X)
print(df.head())
print(df.describe())
# Plotting a single feature to see whether it is normalised
plt.style.use('seaborn')
plt.scatter(X[:, 5], Y)
plt.show()

# Also we need to add a column of ones that is a column for x0 in our matrix X
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))
print(X.shape)

# Implementing multivariate linear regression using vectorization
# Avoid using loops in implementation except gradient descent main loop
# Use numpy functions like np.sum(), np.dot(), etc which are quite fast and already optimised


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


# Creating the function for gradient descent
def gradientDescent(X, Y, max_iters=300, alpha=0.1):
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


# Getting the parameters for our training data
start = time.time()
theta, cost_function_values, theta_values = gradientDescent(X, Y)
end = time.time()
print('Time Taken is: ', end-start)
print("Parameters: ")
print(theta)
print("Decrease in Cost function with each iteration of gradient descent: ")
for cost_function_value in cost_function_values:
    print(cost_function_value)

# Visualizing decrease in Cost function with each iteration of gradient descent
plt.plot(cost_function_values)
plt.xlabel('Number of iteration')
plt.ylabel('Cost function')
plt.title('Decrease in Cost function with each iteration of gradient descent')
plt.show()

# Getting predictions for the test data
predictions = []
for i in range(X.shape[0]):
    pred = hypothesis(X[i], theta)
    predictions.append(pred)
predictions = np.array(predictions)
print(predictions.shape)

# Storing the predictions in a CSV file
# Converting pred into a pandas dataframe
df = pd.DataFrame(predictions, columns=['y'])
# Saving to a CSV file
df.to_csv('predictions.csv', index=False)


# Finding out the accuracy of our model
# To measure the accuracy we use a parameter called as
# R2(R Squared) or Coefficient of Determination
# It is a value between 0 to 1 both inclusive
# To read more about it or to know its formula search online about it
# Creating the function to compute the value of R2
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
