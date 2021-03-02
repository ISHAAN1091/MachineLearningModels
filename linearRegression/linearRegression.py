import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# If you want a good understanding gradient descent or linear regeression you can watch Andrew Ng's course on
# coursera titled "Introduction to Machine Learning By Stanford University"

# Implementing linear regression

# Loading and visualizing the data
# -- 1. Load the data and convert into numpy arrays
# -- 2. Normalize the data
# -- 3. Visualize the data

# Loading the data
X = pd.read_csv('./data/Linear_X_Train.csv')
Y = pd.read_csv('./data/Linear_Y_Train.csv')
X_test = pd.read_csv('./data/Linear_X_Test.csv')
print(X.shape, Y.shape, X_test.shape)
# Converting above data into numpy arrays
X = X.values
Y = Y.values
X_test = X_test.values
print(type(X), type(Y), type(X_test))

# Normalizing the data
u = X.mean()
std = X.std()
X = (X-u)/std

# Visualizing the data
plt.style.use('seaborn')
plt.scatter(X, Y)
plt.xlabel('Hardwork Time')
plt.ylabel('Performance')
plt.title("Hardwork Time vs Performance")
plt.show()


# Creating the hypothesis function
def hypothesis(x, theta):
    # theta = [theta0 theta1] theta0 and theta1 as we know are our parameters
    h_theta = theta[0] + theta[1]*x
    return h_theta


# Creating the function to find gradient
def gradient(X, Y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        h_theta = hypothesis(x, theta)
        y = Y[i]
        grad[0] += h_theta - y
        grad[1] += (h_theta - y)*x
    grad /= m
    return grad


# Creating the cost function
def costFunction(X, Y, theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        x = X[i]
        h_theta = hypothesis(x, theta)
        y = Y[i]
        total_error += (h_theta - y)**2
    total_error /= m
    return total_error


# Creating the function for gradient descent
def gradientDescent(X, Y, max_iters=100, alpha=0.1):
    theta = np.zeros((2,))
    cost_function_values = []
    for i in range(max_iters):
        # Computing the gradient
        grad = gradient(X, Y, theta)
        # Also calculating and storing cost function for each iteration in cost_function_values
        cost_function_value = costFunction(X, Y, theta)
        cost_function_values.append(cost_function_value)
        # Updating theta to a better value to minimise cost function
        theta[0] = theta[0] - alpha*grad[0]
        theta[1] = theta[1] - alpha*grad[1]
    return theta, cost_function_values


# Getting the parameters for our training data
theta, cost_function_values = gradientDescent(X, Y)
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

# Visualizing our training data along with our best fit line
# Getting h_theta for all values of x
h_theta = hypothesis(X, theta)
# Plotting the data and the line
plt.scatter(X, Y)
plt.plot(X, h_theta, color='orange', label='Prediction')
plt.legend()
plt.show()

# Getting predictions for the test data
pred = hypothesis(X_test, theta)
print(pred.shape)

# Storing the predictions in a CSV file
# Converting pred into a pandas dataframe
df = pd.DataFrame(pred, columns=['y'])
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
r2_score = r2Score(Y, h_theta)
print(r2_score)


# Above in calculating minimum value of cost function in gradient descent functionyou can also
# add an automatic break point if the change in cost function gets less than a certain
# value for example 0.001(decide according to the level of accuracy required) . This will help
# you to reduce the number of iterations.
