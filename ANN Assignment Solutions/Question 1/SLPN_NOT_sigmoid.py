"""
List of Perceptron variables-

Given ones:
    Learning rate, eta
    Number of iteration, num_it
    Inputs ,x-(m), +1
    Desired output ,d

To calculate:
    Weights ,w-(m), Bias ,b
    Output ,y
    Local induced field, v
    Threshold/Activation function, sgn()
    Errors/misclassifications, errors
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp

title='Logic NOT'

def Sigma(w,x,b):
    s1 = np.dot(w,x)
    s1 += b
    return s1

def Activation(x):
    return 1.0/(1.0+exp(-x))


def Differential(x):
    y=Activation(x)
    return y*(1-y)

class Perceptron:
    def __init__(self, eta, num_it):
        
        #Initialisations
        self.eta = eta
        self.num_it = num_it
        return
    
    def Fit(self, X, d):
        # X ->[samples,features] ; y ->[samples]
        
        #Initialisations
        self.b = 0
        self.w = 0
        self.errors = []
        
        #Epochs
        for i in range(self.num_it):
            error = 0
            for xi,di in zip(X,d):
                v = Sigma(self.w, xi, self.b)    
                y = Activation(v)
                self.w += (self.eta * (di - y) * xi)
                self.b += (self.eta * (di - y))
                error += (di - y) * xi
                self.errors.append(error)
                
        return self
    
    def Plot_err(self):
        #plt.ylim([-1,1])
        plt.xlim([0,100])
        plt.plot(self.errors)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')
        plt.title(title)
        plt.show()
        return self
    
    def Predict(self, xi):
        v = Sigma(self.w, xi, self.b)    
        y = round(Activation(v))
        return y

    def Plot_ip(self, X, d):
        x1,y1,x2,y2=[],[],[],[]
        for xi,di in zip(X,d):
            if(di==0):
                x1.append(xi), y1.append(xi)
            else:
                x2.append(xi), y2.append(xi)
        plt.xlim([-1,2]), plt.ylim([-1,2])
        plt.scatter(x1, y1, color='blue',label='Logic 0')
        plt.scatter(x2, y2, color='red',label='Logic 1')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title(title)
        plt.legend(loc='upper left')
        plt.show()

    def predict(self, X):
        Y = []
        for xi in X:
            v = Sigma(self.w, xi, self.b)    
            y = round(Activation(v))
            Y.append(y)
        return np.array(Y)
    
    def Decision_boundary(self, X, y):
        h = .02 
        x_min, x_max = X.min() - 1, X.max() + 1
        y_min, y_max = X.min() - 1, X.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        npc = (np.c_[xx.ravel(), yy.ravel()])[:,0]
        Z = self.predict(npc).reshape(xx.shape)
        plt.xlim([-1,2]),plt.ylim([-1,2])
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X, X, c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Input 1'), plt.ylabel('Input 2')
        plt.xlim(xx.min(), xx.max()), plt.ylim(yy.min(), yy.max())
        plt.xticks(()),plt.yticks(())
        plt.legend(loc='best')
        plt.title(title)
        plt.show()
        return

def Conf_mat(tn, tp, fn, fp):
    print("Confusion Matrix")
    print("\t\tPredicted NO\t|Predicted YES")
    print("\t\t----------------|-------------")
    print("Actual NO\t|{}\t\t|{}".format(tn,fp))
    print("Actual YES\t|{}\t\t|{}".format(fn,tp))

ip, op =[], []
for i in range(2):
    ip.append(i)
    op.append(1-i)

X = np.array(ip)
d = np.array(op)

print("Training Data")
for a,b in zip(X,d):
    print(str(a)+'->'+str(b))
    
slpn = Perceptron(0.05,10000)
slpn.Fit(X, d)
print("----------------------------\nWeights\n----------------------------")
print("Bias: {}".format(slpn.b))
print("Weight {}: {}".format(1,slpn.w))
print("----------------------------")
slpn.Plot_ip(X,d)
slpn.Plot_err()
slpn.Decision_boundary(X,d)

A_correct = np.zeros((400,2))
A_false = np.zeros((400,2))
tn = tp = fn = fp = 0
for i in range(100):
    for j in range(2):
        cor_op=1-j
        if(j==1):
            if(slpn.Predict(j) == cor_op):
                A_correct[tn] = np.array(j)
                tn += 1
            else:
                A_false[fp] = np.array(j)
                fp += 1
        else:
            if(slpn.Predict(j) == cor_op):
                A_correct[tn+tp] = np.array(j)
                tp += 1
            else:
                A_false[fp+fn] = np.array(j)
                fn += 1
    
Conf_mat(tn, tp, fn, fp)
acc = 100*((tp+tn)/(tp+tn+fn+fp))
print("Accuracy: "+str(acc)+"%")