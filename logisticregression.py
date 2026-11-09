#working prediction model using logistic regression to predict if things are yes or no if that makes sense based
#on the data giving below, you can add more features if you want

import numpy as np

inputdata = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([0, 0, 1, 0, 1], dtype=float)
m = len(y)

X = np.column_stack([np.ones(m), inputdata])

theta = np.zeros(2)

alpha = 0.9236
#this was the best learning rate i could get to get theta in the least amount of steps


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(y, theta, X, m):
    z = X @ theta
    #it computes theta.T times X
    h = sigmoid(z)
    #gets the sigmoid
    first_term = y @ np.log(h)
    #computes the first part of the logistical regression cost function
    second_term = (1 - y) @ np.log(1 - h)
    #computes the second part of the logistical regression cost function
    return -(first_term + second_term) / m
    # returns J, lowest cost


def gradientdescent(X, y, alpha, theta):
    m = len(y)
    for i in range(5000):
        z = X @ theta
        #computes z for sigmoid so that
        h = sigmoid(z)
        #gets sigmoid
        error = h - y
        #gets the error that is inside the sums thing
        sums = error @ X
        #mults the error by the X that came from doing derivative 
        gradient = sums / m
        #gets the first part of the gradient
        temp = theta - alpha * gradient
        #gets the final temp thetas 
        if np.max(np.abs(temp - theta)) < 1e-9:
            #if the change bettween the current theta and new theta is bearly anything, break and return theta
            return temp
        theta = temp
    return theta

    
theta = gradientdescent(X, y, alpha, theta)
print("best thetas are:", theta)

J = cost(y, theta, X, m)
print("the cost is:", J)


while True:
    userval = float(input("Give an x value to predict: "))
    z = theta[0] + theta[1] * userval
    probablitiy = sigmoid(z)
    print(probablitiy)
    if probablitiy >= .5:
        print("1")
    else:
        print("0")
    




