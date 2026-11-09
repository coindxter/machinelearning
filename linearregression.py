#working prediction model based on linear regression using the normal equation to find theta. you can add more features if you want
#this uses noraml equation, use normal equation unless you have over 1000's of features then use gradient descent


import numpy as np

x1 = np.array([0, 1 ,2, 3], dtype=float) # or you can do np.array([[3, 4], [4, 6]]) to add more features
y = np.array([7, 5, 4, 1])
m = len(y)

X = np.column_stack([np.ones(m), x1])
#adds a column of 1's in the first column for matrice multiplication

def normalequation(X,y):
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta
    #normal function instead of gradient descent bc this is much easier to code
    #theta[0] is theta_0 in the equation (b) in y=mx+b
    #theta[1] is theta_1 in the equation (m) in y=mx+b 
    #print("normal equation:", theta)
  

def costfunction(theta):
    #returns the smallest value J. score for how well the line fits
    #bigger is worse
    error = X @ theta - y
    total = (error ** 2).sum() / (2 * m)
    return total

theta = normalequation(X,y)
print(theta)
J = costfunction(theta)
print("Lowest J is:", J)

while True:
    userval = float(input("Give an x value to predict: "))
    h = theta[0] + theta[1] * userval
    print(f"The predicted value for {userval} is {h:.3}")
    #print(h)

    