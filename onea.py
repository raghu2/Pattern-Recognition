import numpy as np
import matplotlib.pyplot as plt

mean1 = [1,2,3]
cov1 = [[1,2,0],[2,1,3],[0,3,1]]
mean2 = [4,5,6]
cov2 = [[1,2,0],[2,1,3],[0,3,1]]

x1 = np.random.multivariate_normal(mean1, cov1, 100)
y1 = np.random.multivariate_normal(mean1, cov1, 100)
#x2 = np.random.multivariate_normal(mean2, cov2, 100)
#y2 = np.random.multivariate_normal(mean2, cov2, 100)

plt.scatter(x1, y1, color="red")
#plt.scatter(x2, y2, color="blue")
plt.show()

