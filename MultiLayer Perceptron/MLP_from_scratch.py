"""
MLP attempt #3
Given variables:
    num_ip = 4 (Number of features)
    num_op = 3 (Number of classes/labels)
    num_hidden = 3 (Fixed)
    eta = Learning rate (Variable)
    num_epochs = Number of epochs/iterations (Variable)
    tdata = Training data [#Samples, #Features]
    ttarget = Training target [#Samples, #num_op]
"""
import numpy as np
from math import exp
from sklearn import datasets

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
        
        
def Activate(x):
    return 1.0/(1.0 + exp(-x))

def Diff(x):
    y = Activate(x)
    return y * (1 - y)

class MLP:
    
    def __init__(self, eta, num_epochs):
        
        self.eta = eta
        self.num_epochs = num_epochs
        
    def Forwardprop(self, xi):
        for i in range(len(self.v1)):
            self.v1[i] = (np.dot(np.transpose(self.w1)[i], xi))
        for i in range(len(self.y1)):
            self.y1[i] = (Activate(self.v1[i])) 
        for i in range(len(self.v2)):
            self.v2[i] = (np.dot(np.transpose(self.w2)[i], self.y1)) 
        for i in range(len(self.y2)):
            self.y2[i] = ((Activate(self.v2[i])))
        return self.y2
        
    def Backwardprop(self, xi, di):
        self.errors = (di - self.y2)
        self.cost += sum([i**2 for i in self.errors])
        for i in range(len(self.y2)):
            self.G2[i] = (self.errors[i] * Diff(self.v2[i]))
        
        for i in range(len(self.y1)):
            s = 0
            for j in range(len(self.y2)):
                s += (self.G2[j] * self.w2[i][j])
            self.G1[i] = s * (Diff(self.v1[i]))
            
        for i in range(4):
            for j in range(len(self.y1)):
                self.w1[i][j] += (self.eta) * (self.G1[j]) * (xi[i])
        
        for i in range(len(self.y1)):
            for j in range(len(self.y2)):
                self.w2[i][j] += (self.eta) * (self.G2[j]) * (self.y1[i])
        return
    
    def Fit(self, tdata, ttarget):
        #Weights input -> hidden layer
        self.w1 = np.zeros((4, 3))
        #Weights hidden -> output layer
        self.w2 = np.zeros((3, 3))
        for i in range(4):
            for j in range(3):
                self.w1[i][j] = np.random.uniform(-0.5, 0.5)
                
        for i in range(3):
            for j in range(3):
                self.w2[i][j] = np.random.uniform(-0.5, 0.5)
        self.v1 = np.zeros((3))
        self.y1 = np.zeros((3))
        self.v2 = np.zeros((3))
        self.y2 = np.zeros((3))
        self.G1 = np.zeros(len(self.y1))
        self.G2 = np.zeros(len(self.y2))
        #self.errors = []
        
        for epoch in range(self.num_epochs):
            self.cost = 0
            for xi, di in zip(tdata, ttarget):
                self.Forwardprop(xi)
                self.Backwardprop(xi, di)
            print("Epoch: {}\tCost: {}".format(epoch, (self.cost/tdata.shape[0])))
        return
        
    def Predict(self, x):
        for i in range(len(self.v1)):
            self.v1[i] = (np.dot(np.transpose(self.w1)[i], xi))
        for i in range(len(self.y1)):
            self.y1[i] = (Activate(self.v1[i])) 
        for i in range(len(self.v2)):
            self.v2[i] = (np.dot(np.transpose(self.w2)[i], self.y1)) 
        for i in range(len(self.y2)):
            self.y2[i] = (round(Activate(self.v2[i])))
        return argmax(self.y2)
        
iris = datasets.load_iris()
train_data = iris.data[:20, :]
train_target = iris.target[:20]
train_data = np.concatenate((train_data, iris.data[50:70, :]), axis=0)
train_target = np.concatenate((train_target, iris.target[50:70]), axis=0)
train_data = np.concatenate((train_data, iris.data[100:120, :]), axis=0)
train_target = np.concatenate((train_target, iris.target[100:120]), axis=0)
tt = train_target
train_target = onehotvector(train_target)
mlp = MLP(0.4, 1000)
mlp.Fit(train_data, train_target)
print("Training data\n------------")
print("S_w\tS_l\tP_w\tP_l\tPredicted\tExpected")
print("---\t---\t---\t--\t---------\t-------")
corr_pred = corr_total = 0
for xi,di in zip(train_data, tt):
    print("{}\t{}\t{}\t{}\t{}\t{}".format(xi[0], xi[1], xi[2], xi[3], mlp.Predict(xi), di))
    if(mlp.Predict(xi) == di):
        corr_pred += 1
    corr_total += 1
    
acc_train = (corr_pred / corr_total ) * 100.0
test_data = iris.data[20:50, :]
test_target = iris.target[20:50]
test_data = np.concatenate((test_data, iris.data[70:100, :]), axis=0)
test_target = np.concatenate((test_target, iris.target[70:100]), axis=0)
test_data = np.concatenate((test_data, iris.data[120:150, :]), axis=0)
test_target = np.concatenate((test_target, iris.target[120:150]), axis=0)
print("Testing data\n------------")
print("S_w\tS_l\tP_w\tP_l\tPredicted\tExpected")
print("---\t---\t---\t--\t---------\t-------")
corr_pred = corr_total = 0
for xi,di in zip(test_data, test_target):
    print("{}\t{}\t{}\t{}\t{}\t{}".format(xi[0], xi[1], xi[2], xi[3], mlp.Predict(xi), di))
    if(mlp.Predict(xi) == di):
        corr_pred += 1
    corr_total += 1
        
acc_test = (corr_pred / corr_total ) * 100.0
print("Training Accuracy: {}%".format(acc_train))    
print("Testing Accuracy: {}%".format(acc_test))    
