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
# Normalisze the Data

u = X.mean()
std = X.std()
X = (X-u)/std

plt.title("Normalized Data")
plt.scatter(X, Y)
plt.show()


def getW(query_point, X, tau):
    M = X.shape[0]
    W = np.mat(np.eye(M))

    for i in range(M):
        xi = X[i]
        x = query_point
        W[i, i] = np.exp(np.dot((xi-x), (xi-x).T)/(-2*tau*tau))
    return W


X = np.mat(X)
print(X.shape)
Y = np.mat(Y)
M = X.shape[0]


def predict(X, Y, query_x, tau):
    ones = np.ones((M, 1))
    X_ = np.hstack((X, ones))

    qx = np.mat([query_x, 1])

    W = getW(qx, X_, tau)

    # theta = `(X′WX)inv * X′WY`
    theta = np.linalg.pinv(X_.T*(W*X_))*(X_.T*(W*Y))
    # print(theta.shape)
    pred = np.dot(qx, theta)
    return theta, pred


def r2Score(Y, h_theta_):
    # Here to find the sums instead of using a loop np.sum is recommended as it is faster
    numerator = np.sum((h_theta_-Y)**2)
    denominator = np.sum((Y-Y.mean())**2)
    score = 1 - numerator/denominator
    score *= 100
    return score


def plotPrediction(tau):
    X_test = np.linspace(-2, 2, 20)
    print(X_test[:5])
    X_test = np.array(X)
    print(X_test[:5])
    print(type(X_test))
    print(type(X))
    Y_test = []

    for xq in X_test:
        print(type(xq))
        theta, pred = predict(X, Y, xq[0], tau)
        Y_test.append(pred[0][0])

    Y_test = np.array(Y_test)

    XO = np.array(X)
    YO = np.array(Y)

    plt.title("Tau/Bandwidth Param %.2f" % tau)
    plt.scatter(XO, YO)
    plt.scatter(X_test, Y_test, color='red')
    plt.show()
    Y_test = Y_test.reshape((-1, 1))
    YO = YO.reshape((-1, 1))

    # Finding coefficient of determination for our model
    print('Coefficient of Determination: ')
    r2_score = r2Score(YO, Y_test)
    print(r2_score)


taus = [0.1, ]
for t in taus:
    plotPrediction(t)
