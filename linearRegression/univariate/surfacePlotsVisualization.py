import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

# np.meshgrid(arr1,arr2)
# it makes arr1 a matrix of order len(arr2)Xlen(arr1) with each row of the matrix as the array arr1
# and for arr2 it makes a matrix of order len(arr2)Xlen(arr1) with each column of the matrix as the array arr2
# so the two resultant matrices have equal order and if we plot them on a XY-plane taking arr1 as X and arr2
# as Y then we get a mesh like structure So basically meshgrid gives us a plane
a, b = np.meshgrid(a, b)
print(a)
print(b)

# Creating a 3D plot
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_surface(a, b, a+b, cmap='rainbow')
plt.show()

# Creating a 3D parabola type figure or bowl shaped figure
a = np.arange(-1, 1, 0.02)
b = np.arange(-1, 1, 0.02)
a, b = np.meshgrid(a, b)
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_surface(a, b, a**2+b**2, cmap='rainbow')
plt.show()

# Creating a 3D contour plot
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.contour(a, b, a**2+b**2, cmap='rainbow')
plt.show()

# Creating a 2D contour plot
plt.contour(a, b, a**2+b**2, cmap='rainbow')
plt.show()
