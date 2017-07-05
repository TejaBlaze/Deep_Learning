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



def genrand(A2,prev,c_pos):
    c_pos += np.random.randint(0,A2.shape[0])
    for ele in prev:
        if(c_pos == ele):
            flag = 1
            break
        flag = 0
    
    while((not(c_pos>=0 and c_pos<A2.shape[0]))or(flag==1)):
        if(c_pos<0):    
            c_pos += np.random.randint(0,A2.shape[0])
        else:
            c_pos -= np.random.randint(0,A2.shape[0])
        
        for ele in prev:
            if(c_pos == ele):
                flag = 1
                break
            flag = 0
    return c_pos
            
def CreateDS():
    A1 = np.array(np.arange(100))
    A3 = np.array(np.arange(100,200))
    training_data = np.zeros((int(A1.shape[0]/2),3)) 
    i = 0
    prev = []
    c_pos = np.random.randint(0,A3.shape[0])
    prev.append(c_pos)
    for l in range(A1.shape[0]):
        a1 = np.random.choice([-1,1])
        if(a1 == 1):
            training_data[l] = np.array([A1[genrand(A1,prev,c_pos)],A1[genrand(A1,prev,c_pos)],-1])
        else:
            training_data[l] = np.array([A3[genrand(A1,prev,c_pos)],A3[genrand(A1,prev,c_pos)],1])
        i = i+1
        if(i >= len(A1)/2):
            break
    td = []
    for ele in training_data:
        if(ele.all() == 0):
            break
        td.append(ele)
    return np.array(td)



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
        
    def plot_data(self,X):
        A1 = np.zeros((X.shape[0],2))
        A2 = np.zeros((X.shape[0],2))
        i=j=0
        for ele in X:
            if(ele[2] == 1):
                A1[i] = np.array(ele[:2])
                i = i+1
            else:
                A2[j] = np.array(ele[:2])
                j = j+1
        X1 = []
        X2 = []
        for ele in A1:
            if(ele.all() == 0):
                break
            X1.append(ele)
        for ele in A2:
            if(ele.all() == 0):
                break
            X2.append(ele)
        A1 = np.array(X1)
        A2 = np.array(X2)
        print(A1)
        print(A2)
        plt.scatter(A1[:,0],A1[:,1],color='red',marker='x',label='class1')
        plt.scatter(A2[:,0],A2[:,1],color='blue',marker='o',label='class2')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='upper left')
        plt.show()


A = CreateDS()  
print(A)          
X = np.array(A[:,:2])
d = np.array(A[:,2])

print("Training Data")
for a,b in zip(X,d):
    print(str(a)+'     ->     '+str(b))
    
Intclass = Perceptron(0.1, 1000 , X)
Intclass.plot_data(A)
Intclass.Fit(X, d)
Intclass.Visualise()
tn = tp = fn = fp = 0
j = 0
A_correct = np.zeros((int(2e4),2))
A_false = np.zeros((int(2e4),2))

print(Intclass.Predict([0,10]))
print(Intclass.Predict([110,120]))
#print("X\tY\tPredicted\tExpected")
for i in range(100):
    for k in range(100):
        #print(str(i)+"\t"+str(k)+"\t"+str(Intclass.Predict([i,k]))+"\t0")
        if(Intclass.Predict([i,k]) == 0):
            A_correct[tn] = np.array([i,k])
            tn += 1
        else:
            A_false[fp] = ([i,k])
            fp += 1

for i in range(100,200):
    for k in range(100,200):
        #print(str(i)+"\t"+str(k)+"\t"+str(Intclass.Predict([i,k]))+"\t1")
        if(Intclass.Predict([i,k]) == 1 ):
            A_correct[tn+tp] = np.array([i,k])
            tp += 1
        else:
            A_false[fp+fn] = ([i,k])
            fn += 1
acc = 100*((tp+tn)/(tp+tn+fn+fp))
"""
while(acc<75):
    Intclass.Fit(X, d)
    tn = tp = fn = fp = 0
    for i in range(100):
        for k in range(100):
            if(Intclass.Predict([i,k]) == 0):
                A_correct[tn] = np.array([i,k])
                tn += 1
            else:
                A_false[fp] = ([i,k])
                fp += 1
    
    for i in range(100,200):
        for k in range(100,200):
            if(Intclass.Predict([i,k]) == 1 ):
                A_correct[tn+tp] = np.array([i,k])
                tp += 1
            else:
                A_false[fp+fn] = ([i,k])
                fn += 1
    acc = 100*((tp+tn)/(tp+tn+fn+fp))
"""    
x1 = np.array(A_correct[:,0])
y1 = np.array(A_correct[:,1])
x2 = np.array(A_false[:,0])
y2 = np.array(A_false[:,1])
plt.ylim([0,200])
plt.xlim([0,200])
plt.scatter(x1, y1, color='green',label='correct')
plt.scatter(x2, y2, color='red',label='wrong')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='upper left')
plt.show()

print("\n\nConfusion Matrix")
print(['           ','Predicted NO','Predicted YES'])
print(['Actual NO', 'True Negative='+str(tn) , 'False positive='+str(fp)])
print(['Actual YES', 'False Negative='+str(fn), 'True Positive='+str(tp)])

    
print("Accuracy: "+str(acc)+"%")
misacc = 100*((fp+fn)/(tp+tn+fn+fp))
print("Misclassification rate: "+str(misacc)+"%")
tprate = 100*((tp)/(1e4))
print("True Positive rate: "+str(tprate)+"%")