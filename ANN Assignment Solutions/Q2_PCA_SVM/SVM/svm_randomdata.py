from random import uniform, randint
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
def GenerateData(n, ll, ul):
    prev_gen = []
    train_data, train_targets = [],[]
    i=0
    while i<n:
        x= uniform(ll, ul)
        if(i<int(n/2)):
            y = uniform(ll**2,x**2-1)
            cl=0
        else:
            y = uniform(x**2+1,ul**2)
            cl=1    
        flag=0
        if i==0:
            prev_gen.append([x,y])
            train_data.append([x,y]), train_targets.append(cl)
            i+=1
        else:
            for xi,yi in prev_gen:
                if xi==x and yi==y:
                    flag=1
                    break
            if(flag==0):
                prev_gen.append([x,y])
                train_data.append([x,y]), train_targets.append(cl)
                i+=1
    return [train_data, train_targets]
        
def GenerateOutliers(n, ll, ul):
    prev_gen = []
    train_data, train_targets = [],[]
    i=0
    while i<n:
        x= uniform(ll, ul)
        cho = randint(0,100)
        if(cho<50):
            y = uniform(x**2-1, x**2)
            cl=0
        else:
            y = uniform(x**2,x**2+1)
            cl=1    
        flag=0
        if i==0:
            prev_gen.append([x,y])
            train_data.append([x,y]), train_targets.append(cl)
            i+=1
        else:
            for xi,yi in prev_gen:
                if xi==x and yi==y:
                    flag=1
                    break
            if(flag==0):
                prev_gen.append([x,y])
                train_data.append([x,y]), train_targets.append(cl)
                i+=1
    return [train_data, train_targets]
  
def PlotData(train_data, train_targets):
    x1,y1,x2,y2=[],[],[],[]
    for xi,di in zip(train_data, train_targets):
        if di==0:
            x1.append(xi[0]), y1.append(xi[1])
        else:
            x2.append(xi[0]), y2.append(xi[1])
    plt.scatter(x1,y1,color='red')
    plt.scatter(x2,y2,color='blue')
    plt.show()

def Decision_boundary(clf, X, y):
    h = .02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    title = 'Classification with SVM'
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    svs = clf.support_vectors_
    plt.scatter(svs[:,0],svs[:,1],c='black', marker='+')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(()),plt.yticks(())
    plt.title(title)
    plt.show()
    return

train_data, train_targets = GenerateData(50, 0, 10)
print("{} {}".format(len(train_data), len(train_data[0])))
td, tt = train_data, train_targets
for kern in ['linear', 'poly', 'rbf']:
    clf = svm.SVC(kernel=kern, degree=2)
    train_data, train_targets = td, tt
    clf.fit(train_data, train_targets)
    plt.ion()
    for i in range(5):
        axes = plt.gca()
        axes.set_xlim([-1,10])
        axes.set_ylim([-10,105])
        Decision_boundary(clf, np.array(train_data), np.array(train_targets)) 
        out_data, out_targets = GenerateOutliers(1, 0, 10)
        for xi, di in zip(out_data, out_targets):
            train_data.append(xi), train_targets.append(di)
        clf.fit(train_data, train_targets)
        plt.pause(0.05)
        plt.clf()