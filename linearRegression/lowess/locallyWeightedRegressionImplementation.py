# Locally weighted regression
# 1. Read and normalise the dataset
# 2. Generate W for every query point
# 3. No training is involved, directly make predictions using closed form solution we derived -
# inverse(X'WX)*(XWY)
# 4. Find the best value of Tau(Bandwidth parameter) [Cross validation]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading data
dfx = pd.read_csv('./data/weightedX.csv')
dfy = pd.read_csv('./data/weightedY.csv')
# Converting data into numpy arrays
X = dfx.values
Y = dfy.values
print(X.shape, Y.shape)

# Normalising the data
u = X.mean()
std = X.std()
X = (X-u)/std

# Visualizing the data
plt.style.use('seaborn')
plt.scatter(X, Y)
plt.title('Normalised data')
plt.show()


# Finding the value of W matrix
def getW(query_point, X, tau):
    m = X.shape[0]
    W = np.mat(np.eye(m))
    for i in range(m):
        x = query_point
        xi = X[i]
        W[i, i] = np.exp(np.dot((xi-x), (xi-x).T)/(-2*tau*tau))
    return W


# Creating a function to make predictions on our test data
def predict(X, Y, query_point, tau):
    ones = np.ones((X.shape[0], 1))
    X_ = np.hstack((ones, X))
    query_point = np.mat([1, query_point])
    W = getW(query_point, X, tau)
    # Implementing closed form solution to get theta for this specific query point
    theta = np.linalg.pinv(X_.T*(W*X_))*(X_.T*(W*Y))
    pred = np.dot(query_point, theta)
    return theta, pred


# Visualizing predictions and analyzing the effect of tau
def plotPrediction(tau):
    X_test = np.linspace(-2, 2, 20)
    Y_test = []
    for xq in X_test:
        theta, pred = predict(X, Y, xq, tau)
        Y_test.append(pred)
    Y_test = np.array(Y_test)


predictions = []
for i in range(X.shape[0]):
    theta, pred = predict(X, Y, X[i][0], 1.5)
    predictions.append(pred[0][0])
    print(type(X[i][0]))
predictions = np.array(predictions)
predictions = predictions.reshape((-1, 1))

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
