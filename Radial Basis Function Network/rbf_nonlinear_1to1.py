"""
RBFN Version 3(Function approximation):
     Given variables:
        Number of samples: N
        Number of clusters: K
        Training Data: X [#samples, #features]
        Training Targets: d [#samples]
        Regularizing paramter: A
        
    To find:
        Cluster centers: V [K]
        Cluster constituent points: Atok [K, max->N]
        Beta values: B [K]
        Weights vector: W [K, c]
        Green's matrix: G [K, N]
"""
from math import exp, sin, pi
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.cluster import KMeans
from scipy.linalg import pinv
import pandas.read_excel as rdex

def Dist(p1, p2):
    s = 0
    return abs(p1 - p2)
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
        G = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                G[i][j] = exp(-self.B * Dist( X[j], X[i]))
        return G
    
    def Fit(self, X, d):
        self.X = X
        G = self.calcG(X).T
        P = pinv(np.dot(G.T, G) + self.A * np.identity(X.shape[0]))
        Q = np.dot(P, G.T)
        self.w = np.dot(Q, d)
        return
    
    def Predict(self, x):
        G = np.zeros((len(X)))
        for i in range(len(X)):
            G[i] = exp(-self.B * Dist(x, self.X[i]))
        
        return np.dot(G, self.w)
            
    def Funct_approx(self, x, y):
        plt.plot(x, y, color='green', label='Exact function')
        yp = []
        for xi in x:
            yp.append(self.Predict(xi))
        plt.plot(x, yp, color='blue', label='Interpolated function')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Non-linear function approximation')
        plt.legend(loc='best')
        plt.show()
        return

        
       
X = np.arange(-1, 1, 0.1)
d = np.array([(sin(4*pi*xi)*exp(-1*abs(5*xi))) for xi in X])
r1 = rbfn(0.001,1)
r1.Fit(X, d)
r1.Funct_approx(X, d)