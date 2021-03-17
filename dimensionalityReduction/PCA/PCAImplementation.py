from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import svd

# Loading the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
X = X_test.reshape(-1, 28*28)
Y = Y_test

# Normalising the data
sc = StandardScaler()
X_ = sc.fit_transform(X)
print(X_.shape)

# Implementing our own custom PCA
# Calculating covariance
# To calculate the covariance matrix we use the formula X_.T*X_ where * is matrix multiplication
covar = np.dot(X_.T, X_)
print(covar.shape)

# Computing eigen vectors using svd
U, S, V = svd(covar)
print(U.shape)
# Taking only the first two eigenvectors as we need only two dimensions after reduction
U_reduced = U[:, :2]
print(U_reduced.shape)

# Projection of data on new axis(Components)
z = np.dot(X_, U_reduced)
print(z.shape)

# Visualizing the data
new_dataset = np.hstack((z, Y.reshape(-1, 1)))
dataframe = pd.DataFrame(new_dataset, columns=['PC1', 'PC2', 'label'])
print(dataframe.head())

plt.figure(figsize=(15, 15))
fg = sns.FacetGrid(dataframe, hue='label', height=10)
fg.map(plt.scatter, 'PC1', 'PC2')
fg.add_legend()
plt.show()
