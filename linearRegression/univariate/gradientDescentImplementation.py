import numpy as np
import matplotlib.pyplot as plt

# If you want a good understanding gradient descent or linear regeression you can watch Andrew Ng's course on
# coursera titled "Introduction to Machine Learning By Stanford University"

# Implementation of gradient descent
# Given a function f(x) our goal is to find the value of x ,that minimises it ,using gradient descent

# Defining x and a corresponding function y
X = np.arange(10)
Y = (X-5)**2
print(X, Y)

# Visualizing
plt.style.use('seaborn')
plt.plot(X, Y)
plt.ylabel('F(x)')
plt.xlabel('x')
plt.show()
# We note that in the plot our minima is at 5 and minimum value of F(x) at that point is 0

# Taking some initial value of x from which we will start and go downhill(in the direction of slope)
x = 0
# Also defining alpha or the learning rate with which our x will approach the minima
alpha = 0.1

# Taking 50 steps in the downhill direction
plt.plot(X, Y)
for i in range(50):
    # Defining the gradient/slope for F(x) (This is nothing but just the differentiation of Y)
    grad = 2*(x-5)
    # Computing the new value of x
    x = x - alpha*grad
    y = (x-5)**2
    plt.scatter(x, y)
    print(x)

# Visualizing gradient descent
plt.show()

# So we see using gradient descent we reach very close to the minima of F(x) i.e. for x=5

# Above you can also add an automatic break point if the change in y gets less than a certain
# value for example 0.001(decide according to the level of accuracy required) . This will help
# you to avoid defining the number of iterations manually.
