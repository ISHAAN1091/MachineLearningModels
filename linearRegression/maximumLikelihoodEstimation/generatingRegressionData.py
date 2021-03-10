import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Here we try to generate regression data on our own without using any dataset

X = np.arange(20)
print(X)

theta = np.array([2, 3])

noise = 3*np.random.randn(20)
Y_ideal = theta[1]*X + theta[0]
Y_real = theta[1]*X + theta[0] + noise

plt.style.use('seaborn')
plt.plot(X, Y_ideal, color='orange')
plt.scatter(X, Y_real)
plt.scatter(X, noise)
plt.show()
