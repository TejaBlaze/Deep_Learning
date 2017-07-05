"""
RBFN Version 2(Approximate Interpolation):
    Given variables:
        Number of samples: N
        Number of clusters: K
        Training Data: X [#samples, #features]
        Training Targets: d [#samples]
        Number of classes: c
        Regularizing paramter: A
    To find:
        Cluster centers: V [K]
        Cluster constituent points: Atok [K, max->N]
        Beta values: B [K]
        Weights vector: W [K, c]
        Green's matrix: G [K, N]
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt 
from math import exp
from scipy.linalg import pinv
from scipy.interpolate import spline

def onehotvector(x):
    y = []
    for xi in x:
        if(xi == 0):
            y.append([1, 0, 0])
        elif(xi == 1):
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    return np.array(y)    

def argmax(x):
    for i in range(len(x)):
        if(x[i] == 1):
            return i
def Dist(p1, p2):
    s = 0
    for i in range(len(p1)):
        s += (p1[i]-p2[i])**2
    return (s**0.5) 
  
class rbfn:
    
    def __init__(self, A, B):
        self.A = A
        self.B = B
        return
    
    def plot_clusters(self, X):
        plt.scatter(X[:, 0], X[:, 1], marker='o')
        plt.scatter(self.V[:, 0], self.V[:, 1], color='black', marker='x')
        plt.show()
        return
    
    def calcG(self, X):
        G = np.zeros((self.K, X.shape[0]))
        for i in range(self.K):
            for j in range(X.shape[0]):
                G[i][j] = exp(-self.B * Dist(X[j], self.V[i]))
        return G
    
    def Fit(self, X, d, K):
        self.K = K
        km = KMeans(n_clusters = self.K, init='random', n_init=5, max_iter=100).fit(X)
        self.V = km.cluster_centers_
        self.plot_clusters(X)
        G = self.calcG(X).T
        P = pinv(np.dot(G.T, G))
        Q = np.dot(P, G.T)
        self.w = np.dot(Q, d)
        print("Input layer Dimensions: {}".format(X.shape[1]))
        print("RBFN layer Dimensions: {}".format(self.K))
        print("Output layer Dimensions: {} \ncurrent predicted: {}\ncurrent output: {}".format(d.shape[1],np.dot(G, self.w), d))
        print("Weight Dimensions: {} * {} ".format(self.w.shape[0], self.w.shape[1]))
        return
    
    def Predict(self, x):
        G = np.zeros((self.K))
        for i in range(self.K):
            G[i] = exp(-self.B * Dist(x, self.V[i]))
        
        y = np.dot(G, self.w)
        for i in range(len(y)):
            if(y[i] == max(y)):
                return i
            
    def Funct_approx(self, X, d):
        plt.scatter(X[:,0], X[:, 1], color = 'blue', label='Data points')
        
        self.V.view('i8,i8,i8,i8').sort(order=['f0'], axis=0)
        plt.plot(self.V[:,0], self.V[:, 1], label='Interpolated function')
        plt.xlabel('Sepal-width')
        plt.ylabel('Sepal-breadth')
        plt.title('Iris Classification')
        plt.legend(loc='best')
        plt.show()    
        return
    
iris = datasets.load_iris()
r1 = rbfn(40, 1)

train_data = iris.data[:20, :]
train_target = iris.target[:20]
train_data = np.concatenate((train_data, iris.data[50:70, :]))
train_target = np.concatenate((train_target, iris.target[50:70]))
train_data = np.concatenate((train_data, iris.data[100:120, :]))
train_target = onehotvector(np.concatenate((train_target, iris.target[100:120])))

r1.Fit(train_data, train_target, 20)    
    

test_data = iris.data[20:50, :]
test_target = iris.target[20:50]
test_data = np.concatenate((test_data, iris.data[70:100, :]))
test_target = np.concatenate((test_target, iris.target[70:100]))
test_data = np.concatenate((test_data, iris.data[120:, :]))
test_target = np.concatenate((test_target, iris.target[120:]))
corr = 0
tot = 0
for xi,di in zip(test_data, test_target):
    print("Predicted : {}\tExpected : {}".format(r1.Predict(xi), di))
    if(r1.Predict(xi) == di):
        corr += 1
    tot += 1
    
acc = corr/tot
print("\nAccuracy: {}%".format(acc*100))
r1.Funct_approx(test_data, test_target)