import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining style to be used by the matplotlib plots
plt.style.use('seaborn')

# k Nearest Neighbours (kNN)
# In this to predict output for a input we find the k nearest datapoints to that value and
# whatever class the majority of those k points belong to we give that value as the output
# Since we are just finding the nearest points at the time of prediction this does not require
# any training so the training time complexity is O(1)

# Reading data from CSV files
dfx = pd.read_csv('./xdata.csv')
dfy = pd.read_csv('./ydata.csv')
X = dfx.values
Y = dfy.values
# dropping the serial number columns in X and Y
X = X[:, 1:]
Y = Y[:, 1:].reshape((-1,))
print(X.shape)
print(Y.shape)

# Plotting a scatter plot of the data to get a view of the clusters
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

# Taking a point and plotting it on the current dataset scatterplot
query_1 = np.array([2, 3])
query_2 = np.array([4, 5])
query_3 = np.array([0, 1])
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.scatter(query_1[0], query_1[1], color='red')
plt.scatter(query_2[0], query_2[1], color='green')
plt.scatter(query_3[0], query_3[1], color='yellow')
plt.show()


# Creating a function to calculate distance between two given points

def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


# Creating the function for kNN

def knn(X, Y, queryPoint, k=5):
    distVals = []
    m = X.shape[0]  # m is the number of points in the dataset
    # Finding and storing the distance and output of all the points from the required point and storing them
    for i in range(m):
        d = dist(queryPoint, X[i])
        distVals.append((d, Y[i]))
    # Sorting the stored tuples according to the distance and slicing out the k nearest neighbours
    distVals = sorted(distVals)
    distVals = distVals[:k]
    distVals = np.array(distVals)
    # Finding various classes in the minimum k datapoints and finding out the majority class
    countClasses = np.unique(distVals[:, 1], return_counts=True)
    maxClassIndex = countClasses[1].argmax()
    # Finally returning the output of the majority Class as our prediction
    prediction = countClasses[0][maxClassIndex]
    return prediction


# Getting the predictions for query_1, query_2 and query_3
pred1 = knn(X, Y, query_1)
print(pred1)
pred2 = knn(X, Y, query_2)
print(pred2)
pred3 = knn(X, Y, query_3)
print(pred3)
