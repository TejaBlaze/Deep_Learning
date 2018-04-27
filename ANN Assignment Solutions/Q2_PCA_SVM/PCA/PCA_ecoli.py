import csv
from sklearn import svm
from random import randint
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def Load_labels(fn):
    dict={}
    with open(fn, 'rt') as f:
        ln = f.readline().strip()
        while len(ln)>0:
            s1 = ln.split(' ')
            id, label = s1[0], s1[1]
            dict[int(id)] = label
            ln = f.readline().strip()
    return dict

def Load_data(fn):
    t_data, t_targets = [],[]
    with open(fn, 'rt') as csvfile:
        sr = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in sr:
            li = [float(ele) for ele in row[0][:-1].split(',') if ele!='']
            t_data.append(li), t_targets.append(float(row[0][-1]))
    return [t_data, t_targets]

def Plot_ip(X, Y):
    no_class = len(set(Y))
    Xx = []
    for i in range(no_class):
        Xx.append([-100,-100])
    for xi,di in zip(X,Y):
        cl = int(di)
        #print(Xx[cl])
        if Xx[cl][0]==-100 and Xx[cl][1]==-100:
            Xx[cl] = []
            Xx[cl].append([ele for ele in xi])
        else:    
            Xx[cl].append([ele for ele in xi])
    class_dict = Load_labels('Dataset/classmap.txt')
    colors = cm.rainbow(np.linspace(0, 1, no_class))
    for i, c in enumerate(colors):
        xx,yy = [],[]
        for Xxi in Xx[i]:
            xx.append(Xxi[0]),yy.append(Xxi[1])
        plt.scatter(xx, yy, color=c, label=class_dict[i])
    plt.title('Ecoli dataset in 2D')
    plt.legend(loc='best')
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

    
train_data, train_targets = Load_data('Dataset/ecoli_train.csv')
clf = svm.SVC()
clf.fit(train_data, train_targets)
test_data, test_targets = Load_data('Dataset/ecoli_test.csv')

print("Before PCA")
test_preds = clf.predict(test_data)
c=0
for xi,pi,di in zip(test_data,test_preds,test_targets):
    if pi==di:
        c+=1
print("Accuracy: {:.2f}%".format(c*100/len(test_data)))    

no_c = int(input("PCA\nNumber of components: "))
print("After PCA")

pca = PCA(n_components=no_c, svd_solver='full')
new_train_data = pca.fit_transform(train_data)
new_test_data = pca.fit_transform(test_data)


clf = svm.SVC()
clf.fit(new_train_data , train_targets)

if no_c==2:
    Plot_ip(new_train_data , train_targets)
    Decision_boundary(clf, new_train_data, train_targets)

test_preds = clf.predict(new_test_data)
c=0
for xi,pi,di in zip(new_test_data,test_preds,test_targets):
    if pi==di:
        c+=1

print("Accuracy: {:.2f}%".format(c*100/len(new_test_data)))    

#print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  


'''
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)                 
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)
'''