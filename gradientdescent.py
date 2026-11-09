#this is linear regresison using just gradient descent

import numpy as np

x1 = np.array([1, 2, 3, 4, 5], dtype=float)
#x2 = np.array([2, 1, 4, 3, 5], dtype=float)
y = np.array([6, 5, 12, 11, 16], dtype=float)
m = len(y)

X = np.column_stack([np.ones(m), x1, ])


theta = np.zeros(2)
#makes array of beginning theta of 2 zeros 
#theta1 = 0 theta2 = 0

alpha = 0.05
#learning rate, find highest learning rate without failing for balence

def costfunction(theta):
    #returns the smallest value J. score for how well the line fits
    #bigger is worse
    error = X @ theta - y
    total = (error ** 2).sum() / (2 * m)
    print(np.round(total, 6))


def gradientdescent(theta):
    print("gradientdescent")
    for i in range(5000):
        error = X @ theta - y
        sums = X.T @ error
        temp = theta - alpha * 1/len(y) * (sums)
        if np.max(np.abs(temp - theta)) < 1e-9: 
            #checks if the new value of theta is different than the new value
            #with a tolerance of 0.0000000001 of it is isnt different than it breaks the for loop
            print(i, theta)
            costfunction(theta)
            break
        theta = temp
        


gradientdescent(theta)
