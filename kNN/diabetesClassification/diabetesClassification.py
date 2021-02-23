import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# k Nearest Neighbour (kNN)

# Reading the diabetes dataset from the CSV file
X_train = pd.read_csv('./Diabetes_XTrain.csv')
Y_train = pd.read_csv('./Diabetes_YTrain.csv')
X_test = pd.read_csv('./Diabetes_Xtest.csv')
print(X_train.shape, Y_train.shape, X_test.shape)

# Plotting a bar graph between number of classes and number of examples in each class
plt.bar([0, 1], [Y_train.value_counts()[0],
                 Y_train.value_counts()[1]], tick_label=['0', '1'])
plt.xlabel('Classes')
plt.ylabel('Number of examples in that class')
plt.title('Number of classes vs Number of examples in that class')
plt.show()

# Converting above dataframes into numpy arrays
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
print(type(X_train), type(Y_train), type(X_test))

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
    distVals = np.array(distVals, dtype=object)
    # Finding various classes in the minimum k datapoints and finding out the majority class
    countClasses = np.unique(distVals[:, 1], return_counts=True)
    maxClassIndex = countClasses[1].argmax()
    # Finally returning the output of the majority Class as our prediction
    prediction = countClasses[0][maxClassIndex]
    return prediction


# Getting the predictions for test data and storing them in a CSV file
with open('preds.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Outcome'])
for i in range(X_test.shape[0]):
    pred = knn(X_train, Y_train, X_test[i])
    with open('preds.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([pred[0]])
