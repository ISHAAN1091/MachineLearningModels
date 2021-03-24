from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
import matplotlib.pyplot as plt

# Data Preparation
digits = load_digits()
X = digits.data
Y = digits.target
print(X.shape, Y.shape)

# Initialising the models
mnb = MultinomialNB()
gnb = GaussianNB()

# Training the models on the data
mnb.fit(X, Y)
gnb.fit(X, Y)

# Calculating the scores for the models to see which performs better
print(mnb.score(X, Y))
print(gnb.score(X, Y))

# Finding cross_val_score or the average accuracy for the models
print(cross_val_score(mnb, X, Y, scoring='accuracy', cv=10).mean())
print(cross_val_score(gnb, X, Y, scoring='accuracy', cv=10).mean())

# So we observe that multinomial model performs better than the gaussian model
# This is because gaussian model makes much more stronger assumptions as compared
# to the weaker assumptions made by the multinomial model
# Or in simpler terms here each feature is more similar to a discrete value than a continuous distribution
# hence multinomial fits much better than gaussian
