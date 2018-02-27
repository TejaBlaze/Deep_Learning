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

title=''#'Random points'

def Sigma(w,x,b):
    s1 = np.dot(w,x)
    s1 += b
    return s1

def Activation(x):
    
    if(x >= 0.0):
        return 1
    return 0
    
    #return 1.0/(1.0+exp(-x))

def Generate_random():
    ip, op =[], []
    l1,l2=0,0
    while(l1<20):
        i,j = np.random.randint(0,100),np.random.randint(0,100)
        if(i+j < 60):
            l1+=1
            ip.append([i,j])
            op.append(0)
        
    while(l2<20):
        i,j = np.random.randint(0,100),np.random.randint(0,100)
        if(i+j > 120):
            l2+=1
            ip.append([i,j])
            op.append(1)
    tr_data =[]
    for i,o in zip(ip,op):
        tr_data.append([i,o])
    np.random.shuffle(tr_data)
    ip,op =[],[]
    for i,o in tr_data:
        ip.append(i), op.append(o)
    return [ip,op]

def add_Outliers(X,d,n):
    ip, op =[ele for ele in X], [ele for ele in d]
    l1,l2=0,0
    while(l1<n):
        i,j = np.random.randint(0,100),np.random.randint(0,100)
        if(i+j > 80 and i+j < 100):
            l1+=1
            ip.append([i,j])
            op.append(np.random.randint(0,2))
    return [np.array(ip),np.array(op)]
    

def Differential(x):
    y=Activation(x)
    return 1#y*(1-y)

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
        #for i in range(X.shape[1]):
        #    self.w[i] = np.random.uniform(-0.5,0.5)
        self.errors = []
        
        #Epochs
        for i in range(self.num_it):
            error = 0
            for xi,di in zip(X,d):
                v = Sigma(self.w, xi, self.b)    
                y = Activation(v)
                for i in range(len(xi)):
                    self.w[i] += (self.eta * (di - y) *Differential(v)*  xi[i])
                self.b += (self.eta *Differential(v)* (di - y) )
                cost = (di - y)**2 
                self.errors.append(cost)
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
        x1,y1,x2,y2,x3,y3,x4,y4=[],[],[],[],[],[],[],[]
        for xi,di in zip(X,d):
            s=xi[0]+xi[1]
            if(s > 80 and s < 100):
                if(di==0):
                    x3.append(xi[0]), y3.append(xi[1])
                else:
                    x4.append(xi[0]), y4.append(xi[1])
            else:
                if(di==0):
                    x1.append(xi[0]), y1.append(xi[1])
                else:
                    x2.append(xi[0]), y2.append(xi[1])
        #plt.xlim([-1,2]), plt.ylim([-1,2])
        plt.scatter(x1, y1, color='blue',label='Class 0')
        plt.scatter(x2, y2, color='red',label='Class 1')
        plt.scatter(x3, y3, color='blue',label='Class 0',marker='x')
        plt.scatter(x4, y4, color='red',label='Class 1',marker='x')
        plt.title(title)
        #plt.legend(loc='upper left')
        #plt.show()

    def predict(self, X):
        Y = []
        for xi in X:
            v = Sigma(self.w, xi, self.b)    
            y = round(Activation(v))
            Y.append(y)
        return np.array(Y)
    
    def Decbound(self,X,d):
        xx,yy = np.arange(0,100,0.2),np.arange(0,100,0.2)
        x1,y1,x2,y2=[],[],[],[]
        for xi in xx:
            for yi in yy:    
                if(self.Predict([xi,yi]) == 0):
                    x1.append(xi),y1.append(yi)
                else:
                    x2.append(xi),y2.append(yi)
        plt.scatter(x1,y1,color='aquamarine')
        plt.scatter(x2,y2,color='coral')
        self.Plot_ip(X,d)
        #plt.show()

def Conf_mat(tn, tp, fn, fp):
    print("Confusion Matrix")
    print("\t\tPredicted NO\t|Predicted YES")
    print("\t\t----------------|-------------")
    print("Actual NO\t|{}\t\t|{}".format(tn,fp))
    print("Actual YES\t|{}\t\t|{}".format(fn,tp))


slpn = Perceptron(0.1,1000)
ip, op =Generate_random()

X = np.array(ip)
d = np.array(op)
    
slpn.Fit(X, d)
slpn.Decbound(X,d)
plt.title('Without Outliers')
plt.show()
#Second time -> add outliers


for k in range(4):
    plt.subplot(2,2,1+k)
    X,d = add_Outliers(X,d,1)
    slpn.Fit(X, d)
    slpn.Decbound(X,d)
    plt.title(str(k+1)+' Outliers')
    plt.xticks([]),plt.yticks([])

plt.show()
