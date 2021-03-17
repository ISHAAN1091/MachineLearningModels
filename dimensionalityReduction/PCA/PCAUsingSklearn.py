from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Implementing PCA using sklearn
# n_components - used to define the number of components finally required after dimensionality reduction
pca = PCA(n_components=2)
z_pca = pca.fit_transform(X_)
print(z_pca.shape)
# .explained_variance_ gives the percentage of original data represented by each of the variables in z_pca
# that is by each of the variables presesnt after dimensionality reduction
# Also note that the sum might not be 100% for total of all these percentages as some data might also get
# lost in dimensionality reduction
print(pca.explained_variance_)
