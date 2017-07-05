import numpy as np
from sklearn import datasets
from math import exp
def mod(x):
    return sum([i for i in x])

def Predict(w, x):
    return np.dot(w, x)

iris = datasets.load_iris()

X = iris.data[:5, :]
X = np.concatenate((X, iris.data[50:55, :]))

Y = iris.target[:5]
Y = np.concatenate((Y, iris.target[50:55]))

phi = np.zeros((X.shape[0], X.shape[0]))
G = 0.5

for i in range(len(X)):
    for j in range(len(X)):
        phi[i][j] = exp( -G * mod(X[i] - X[j]))
        
c = ct = 0 
W = np.linalg.inv(phi) * Y
for i in range(len(W)):
    for j in range(len(W[i])):
        if(W[i][j] == 0):
            c += 1
        ct += 1

print("Zero count :{}".format(c))

print("Total count :{}".format(ct))



