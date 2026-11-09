import numpy as np
import matplotlib.pyplot as plt



x1 = np.array([1, 2, 1.5, 5, 6, 5.5, 3, 3.5, 2.5 ], dtype=float)
x2 = np.array([1, 1.5, 2, 1, 2, 1.5, 5, 6, 5.5], dtype=float)
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=float)

X = np.column_stack([x1, x2])

classes = np.unique(y)

Y = (y[:, None] == classes).astype(float)

actual_label = ["lemon", "melon", "peach"]

m = len(y)

alpha = .5

#amount of neruons
neruon = 4

#hidden layers
hidden = 2

#creates the thetas like normal, but thetas are now weights and bias
#instead of zeros, start with small random values so that the network actually learns
W1 = np.random.randn(2, neruon) * 0.1
W2 = np.random.randn(neruon, 3) * 0.1

b1 = np.zeros((1, neruon))
b2 = np.zeros((1, 3)) 

def sigmoid(z): return 1 / (1 + np.exp(-z))

for i in range(10000):

    #forward propogation one pass only 
    a1 = sigmoid(X @ W1 + b1)
    Y_hat = sigmoid(a1 @ W2 + b2)

    #backwards propogation

    d2 = Y_hat - Y

    dW2 = a1.T @ d2 / m
    db2 = d2.sum(axis=0, keepdims=True) / m    

    d1 = (d2 @ W2.T) * a1 * (1-a1)

    dW1 = X.T @ d1 / m 
    db1 = d1.sum(axis=0, keepdims=True) / m    


    #gradient descent
    W2 -= alpha * dW2
    W1 -= alpha * dW1

    b2 -= alpha * db2
    b1 -= alpha * db1


def forewardpropogation(x1, x2):
    x = np.array([[x1, x2]])             # (1, 2): one sample, two features
    a1 = sigmoid(x @ W1 + b1)            # (1, 2) hidden layer
    out = sigmoid(a1 @ W2 + b2)          # (1, 3) one score per class
    k = np.argmax(out, axis=1)[0]        # index of highest score
    return actual_label[k], out



print("-" * 30)

"""
while True:
    a = float(input("x1: "))
    b = float(input("x2: "))
        
    label, scores = forewardpropogation(a, b)
    print(f"its probably a: {label}")

"""

#this is to test every thing from 0-10 on both features
for i in range(11):
    for j in range(11):
        label, scores = forewardpropogation(i, j)
        print(f"{i, j} its probably a: {label}")




#this is to visulize the data, i did not put this together, thank you claude.
gx, gy = np.meshgrid(np.linspace(0,10,200), np.linspace(0,10,200))
G = np.column_stack([gx.ravel(), gy.ravel()])
out = sigmoid(sigmoid(G @ W1 + b1) @ W2 + b2)
plt.contourf(gx, gy, np.argmax(out, axis=1).reshape(gx.shape), alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.show()
