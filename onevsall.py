#working prediction model using logistic regression to predict if things are yes or no if that makes sense based
#on the data giving below, you can add more features if you want

import numpy as np

"""
this is classifing fruits
x1 = weight of fruit oz (0-10)
x2 = sweetness (0-10)
class 1 lemon, light and sour
class 2 melon, heavy and little sweet
clsas 3 peach, medium and very sweet

"""

x1 = np.array([1, 2, 1.5, 5, 6, 5.5, 3, 3.5, 2.5 ], dtype=float)
x2 = np.array([1, 1.5, 2, 1, 2, 1.5, 5, 6, 5.5], dtype=float)
# y is classes, class 1, clsas 2, class 3
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=float)

X = np.column_stack([np.ones(len(y)), x1, x2])

classes = np.unique(y)
actual_label = ["lemon", "melon", "peach"]
#gets the unique y values and puts them into a class of itsself, only one of each number

alpha = 0.1
#learning rate / alpha

all_thetas = []
#where all the final thetas go


def sigmoid(z): return 1 / (1 + np.exp(-z))

"""
this isnt used at all, just a sanity check to see if the cost is low. i trust my gradient descent
def cost(y, theta, X):
    m = len(y)
    z = X @ theta
    h = sigmoid(z)
    first_term = y @ np.log(h)
    second_term = (1 - y) @ np.log(1 - h)
    return -(first_term + second_term) / m
"""
    
def gradientdescent(X, y, alpha, theta):
    m = len(y)
    for i in range(5000):
        z = X @ theta
        h = sigmoid(z)
        error = h - y
        sums = error @ X
        gradient = sums / m
        temp = theta - alpha * gradient
        if np.max(np.abs(temp - theta)) < 1e-9:
            return temp
        theta = temp
    return theta




#this is the loop to get the thetas for each line it makes
for k in classes:
    #resets theta to 0 for each class so the class has a starting point
    theta = np.zeros(3)

    #this does two things. first it checks to see if any values in y are equal to any of the class data, then it becomes true, true, false etc, than astype turns those booleans into 1's and 0's
    y_k = (y == k).astype(int)
    """ex.
    #y_k = ([0, 1, 2, 1, 0] == 1).astype(int)
    #y_k = [False, True, False, True, False].astype(int)
    #y_k = [0, 1, 0, 1, 0]
    """
    #computes gradient descent for each classs to get the line
    theta = gradientdescent(X, y_k, alpha, theta)

    #stores thetas into another varaibles so they arn't lost when reset at the beginning of the loop
    all_thetas.append(theta)
    
# turn the list into a 3x3 array
thetas = np.array(all_thetas)        
#print(thetas)

#computes cost of the good thetas that were fround in the previous for loop
for i in range(len(classes)):
    #this does two things. first it checks to see if any values in y are equal to any of the class data, then it becomes true, true, false etc, than astype turns those booleans into 1's and 0's
    y_k = (y == classes[i]).astype(int) # .astype converts an array to a type of data, in this example, int
    """ex.
    #y_k = ([0, 1, 2, 1, 0] == 1).astype(int)
    #y_k = [False, True, False, True, False].astype(int)
    #y_k = [0, 1, 0, 1, 0]
    """
    #J = cost(y_k, thetas[i], X)
    #print(f"class {classes[i]} cost: {J}")


print("x1 is weight in oz. 1 is light 10 is heavy")
print("x2 is sweetness, 1 is bitter 10 is super sweet")

#prediction
while True:
    new_x1 = float(input("Give a x1 value to predict: "))
    new_x2 =float(input("Give a x2 value to predict: "))
    new_X = np.array([1, new_x1, new_x2])
    z = new_X @ thetas.T #gets z for the sigmoid function
    h = sigmoid(z) # computes sigmoid function
    best_class = np.argmax(h) # returns index of the largest value in the h array from sigmoid function
    print(f"its probably a {actual_label[best_class]}")


    
    



