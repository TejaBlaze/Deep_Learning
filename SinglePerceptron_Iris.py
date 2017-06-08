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
from sklearn import datasets

def plot_data(X, d):
    c1 = []
    c2 = []
    for xi, di in zip(X, d):
        if(di == 0):
            c1.append(np.array(xi))
        elif(di == 1):
            c2.append(np.array(xi))
    c1 = np.array(c1)
    c2 = np.array(c2)
    plt.scatter(c1[:,0], c1[:,1], color = 'blue', label = 'Iris-setosa')
    plt.scatter(c2[:,0], c2[:,1], color = 'red', label = 'Iris-versicolor')
    plt.xlabel('Sepal width')
    plt.ylabel('Sepal length')
    plt.legend(loc = "best")
    plt.show()
    
    
def Sigma(w,x,b):
    s1 = np.dot(w,x)
    s1 += b
    return s1

def Activation(x):
    if(x >= 0.0):
        return 1
    return 0

class Perceptron:
    def __init__(self, eta, num_it , X):
        
        #Initialisations
        self.eta = eta
        self.num_it = num_it
        #Initialisations
        self.b = 0
        self.w = np.zeros(X.shape[1])
        self.errors = []
        return
    
    def Fit(self, X, d):
        # X ->[samples,features] ; y ->[samples]
        
        
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


iris = datasets.load_iris()
X = np.array(iris.data[10:25,:2])
d = np.array(iris.target[10:25])
X = np.concatenate((X, iris.data[70:85,:2]),axis = 0)
d = np.concatenate((d, iris.target[70:85]), axis = 0)


#For randomising data--

A = []
for xi, di in zip(X, d):
    A.append(np.array([xi,di]))
np.random.shuffle(A)
i = 0
for ele in A:
    #print(str(ele[0][0])+"\t"+str(ele[0][1])+"\t->\t"+str(ele[1]))
    X[i] = np.array([ele[0][0],ele[0][1]])
    d[i] = ele[1]
    i = i + 1

cs = cv = 0
print("Training data\n------------\nIndex\tS_w\tS_d\tLabel")
for i in range(X.shape[0]):
    print(str(i)+"\t"+str(X[i][0])+"\t"+str(X[i][1])+"\t"+str(d[i]))
    if(d[i] == 0):
        cs += 1
    else:
        cv += 1
print("Count setosa =" +str(cs)+"\nCount Versicolor ="+str(cv))
plot_data(X, d)
    
Intclass = Perceptron(0.1,1000 , X)
#plot_data(X , d)
Intclass.Fit(X, d)
Intclass.Visualise()
tn = tp = fn = fp = 0
j = 0
testing_data = iris.data[:100,:2]
d1 = iris.target[:100]
A_correct = np.zeros((testing_data.shape[0],2))
A_false = np.zeros((testing_data.shape[0],2))
print("\nTest data\n-----------")
print("S_l\tS_w\tPredicted\tExpected")
for ele,di in zip(testing_data,d1):
    if(Intclass.Predict(ele) == 0):
        pr = 0
    else:
        pr = 1
    print(str(ele[0])+"\t"+str(ele[1])+"\t"+str(pr)+"\t"+str(di))
    if(not(ele.all() == 0)):
        if((Intclass.Predict(ele)) == 0):
            if(di == 0):
                A_correct[tn] = np.array(ele)
                tn += 1
            else:
                A_false[fp] = np.array(ele)
                fp += 1
        else:
            if(di == 1):
                A_correct[tn+tp] = np.array(ele)
                tp += 1
            else:
                A_false[fp+fn] = np.array(ele)
                fn += 1

#print(A_correct)
#print(A_false)   

x1 = np.array(A_correct[:,0])
y1 = np.array(A_correct[:,1])
x2 = np.array(A_false[:,0])
y2 = np.array(A_false[:,1])
plt.scatter(x1, y1, color='green',label='correct')
plt.scatter(x2, y2, color='red',label='wrong')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='upper left')
plt.show()

print("\n\nConfusion Matrix\n")
print("\t\t   Predicted NO\tPredicted YES")
print("\t\t   ------------\t-------------")
print("Actual NO \t\t| True Negative="+str(tn)+"\tFalse positive="+str(fp))
print("Actual YES\t\t| False Negative="+str(fn)+"\tTrue Positive="+str(tp))
acc = 100*((tp+tn)/(tp+tn+fn+fp))

    
print("Accuracy: "+str(acc)+"%")
misacc = 100*((fp+fn)/(tp+tn+fn+fp))
print("Misclassification rate: "+str(misacc)+"%")
tprate = 100*((tp)/(50))
print("True Positive rate: "+str(tprate)+"%")
