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



def Sigma(w,x,b):
    s1 = np.dot(w,x)
    s1 += b
    return s1

def Activation(x):
    if(x >= 0.0):
        return 1
    return 0

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
        self.w = np.zeros(X.shape[1])
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
    
    def Visualise(self):
        plt.ylim([-1,1])
        plt.xlim([0,60])
        plt.plot(self.errors)
        plt.xlabel('Number of iterations')
        plt.ylabel('Number of misclassifications')
        plt.show()
        return self
    
    def Predict(self, xi):
        v = Sigma(self.w, xi, self.b)    
        y = Activation(v)
        return y
            
X = np.array([[0,0],[0,1],[1,0],[1,1]])
d = np.array([0,1,1,1])

print("Training Data")
for a,b in zip(X,d):
    print(str(a)+'->'+str(b))
    
OR_AND = Perceptron(0.01,100)
OR_AND.Fit(X, d)
OR_AND.Visualise()

A_correct = np.zeros((400,2))
A_false = np.zeros((400,2))
tn = tp = fn = fp = 0
j = 0
for i in range(100):
    for k in range(2):
        if(OR_AND.Predict([j,k]) == 0 ):#and k==0):
            A_correct[tn] = np.array([i,k])
            tn += 1
        elif(OR_AND.Predict([j,k]) == 1):# and k==0):
            A_false[fp] = ([i,k])
            fp += 1

j = 1
for i in range(100):
    for k in range(2):
        if(OR_AND.Predict([j,k]) == 1 and k==1):
            A_correct[tn+tp] = np.array([i,k])
            tp += 1
        elif(OR_AND.Predict([j,k]) == 0 and k==1):
            A_false[fp+fn] = ([i,k])
            fn += 1
    
x1 = np.array(A_correct[:,0])
y1 = np.array(A_correct[:,1])
x2 = np.array(A_false[:,0])
y2 = np.array(A_false[:,1])
plt.ylim([0,1])
plt.xlim([0,100])
plt.scatter(x1, y1, color='green',label='correct')
plt.scatter(x2, y2, color='red',label='wrong')
plt.xlabel('Number of iterations')
plt.ylabel('Output')
plt.legend(loc='upper left')
plt.show()
   
print("Confusion Matrix")
print(['           ','Predicted NO','Predicted YES'])
print(['Actual NO', 'True Negative='+str(tn) , 'False positive='+str(fp)])
print(['Actual YES', 'False Negative='+str(fn), 'True Positive='+str(tp)])
acc = 100*((tp+tn)/(tp+tn+fn+fp))
print("Accuracy: "+str(acc)+"%")
