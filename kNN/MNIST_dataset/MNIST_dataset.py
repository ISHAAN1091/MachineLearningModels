import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# k Nearest Neighbour (kNN)

# Reading the MNIST dataset from the CSV file
df = pd.read_csv('./mnist_train.csv')
print(df.shape)

# Creating a numpy array of the dataframe
data = df.values
print(type(data))

# Separating out the input and output variables from the dataset
X = data[:, 1:]
Y = data[:, 0]
print(X.shape, Y.shape)

# Splitting the data into testing(20%) and training data(80%)
# Note that even though there is no training in kNN it doesn't mean
# we don't have training data
split = int(0.8*X.shape[0])
print(split)
X_train, Y_train = X[:split, :], Y[:split]
X_test, Y_test = X[split:, :], Y[split:]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Creating a function to visualize the image from the MNIST dataset


def drawImg(X, Y, i):
    img = X[i].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(Y[i])
    plt.grid(False)
    plt.show()

# Now applying kNN
# Note that the kNN function/algo/code we wrote for the previouse
# case/code in main.py, the same would be used here as thereis no change in the implementation
# Just the difference is that there we had data with 2 dimensions here we have 784 dimensions


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


# Getting the predictions for test data
pred1 = knn(X_train, Y_train, X_test[0])
print(pred1)
drawImg(X_test, Y_test, 0)
print(Y_test[0])
