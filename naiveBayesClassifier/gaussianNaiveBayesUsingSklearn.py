from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Creating our dataset
X, Y = make_classification(n_samples=200, n_features=2,
                           n_informative=2, n_redundant=0)
print(X.shape, Y.shape)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

# Initialising our model
gnb = GaussianNB()
# Training our classifier
gnb.fit(X, Y)
# Finding out the score of our model
print(gnb.score(X, Y))
# Getting the predictions
ypred = gnb.predict(X)
print(ypred.shape)
acc = (np.sum(ypred == Y))/X.shape[0]
print(acc)
# So we can see we get the same value for acc and score of the model

# This is how we can use gaussian naive bayes using sklearn
