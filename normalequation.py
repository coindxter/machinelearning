#normal equation to find theta, super simple

import numpy as np

X = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
y = np.array([2, 4, 5, 4, 5])

inputdata = np.array([2104, 1416 ,1534, 852])
y = np.array([460, 232, 315, 178])
X = np.column_stack([np.ones(len(y)), inputdata])




theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

print(theta)
