import numpy
import matplotlib.pyplot as plt

# Set up grid and test data
nx, ny = 256, 1024
x = range(nx)
y = range(ny)

data = numpy.random.random((nx, ny))

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X.T, Y.T, data)
plt.show()