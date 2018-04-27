import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
import csv

def Load_data(fn):
    t_data, t_targets = [],[]
    with open(fn, 'rt') as csvfile:
        sr = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in sr:
            li = [float(ele) for ele in row[0][:-1].split(',') if ele!='']
            t_data.append(li), t_targets.append(float(row[0][-1]))
    return [t_data, t_targets]
    
def Plot_ip(X, y, mk='o'):
    no_class = len(set(y))
    Xx = []
    for i in range(no_class):
        Xx.append([-100,-100])
    for xi,di in zip(X,y):
        cl = int(di)
        if Xx[cl][0]==-100 and Xx[cl][1]==-100:
            Xx[cl] = []
            Xx[cl].append([ele for ele in xi])
        else:    
            Xx[cl].append([ele for ele in xi])
    class_dict = {0:'Iris Setosa', 1:'Iris Versicolour', 2:'Iris Virginica'}
    colors = cm.rainbow(np.linspace(0, 1, no_class))
    for i, c in enumerate(colors):
        xx,yy = [],[]
        for Xxi in Xx[i]:
            xx.append(Xxi[0]),yy.append(Xxi[1])
        plt.scatter(xx, yy, color=c, label=class_dict[i], marker=mk)
    plt.title('Iris dataset in 2D - first '+str(no_class)+' classes')
    plt.legend(loc='best')
    #plt.show()


def Decision_boundary(clf, X, y, title):
    h = .02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('X'), plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(()),plt.yticks(())
    plt.title(title)
    plt.show()
    return


def Knn(X_train, y_train, X_test, y_test, weights='uniform', nn=3):
    neigh = KNeighborsClassifier(n_neighbors=nn, weights=weights)
    neigh.fit(X_train , y_train)
    y_pred = neigh.predict(X_test)
    Evaluate_performance(y_test, y_pred)
    return neigh

def Svm(X_train, y_train, X_test, y_test, kern='rbf'):
    clf = svm.SVC(kernel=kern)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Evaluate_performance(y_test, y_pred)
    return clf

def Evaluate_performance(y_test, y_pred):
    corr = 0
    tp,tn,fp,fn=0,0,0,0
    for yi,di in zip(y_pred, y_test):
        if di==1:
            if yi==di:
                tp+=1
            else:
                fn+=1
        elif di==0:
            if yi==di:
                tn+=1
            else:
                fp+=1
        if yi==di:
            corr+=1
    acc = float(corr/float(len(y_test)))
    print("Confusion Matrix")
    print("-----------------")
    print("\t\t Predicted Yes\t Predicted No")
    print("\t\t -------------\t ------------")
    print("Actual Yes\t | {}\t\t | {}".format(tp, fn))
    print("Actual No\t | {}\t\t | {}".format(fp, tn))
    print("\n\nAccuracy: {:.2f}%\nError rate: {:.2f}%\n\n\n\n".format(acc*100, (1-acc)*100))

#Load the dataset
X_train, y_train = Load_data('Dataset/iris_train.csv')
X_test, y_test = Load_data('Dataset/iris_test.csv')

print("\nBefore PCA\n----------")
#Test KNN before PCA
print("\n\nK-Nearest Neigbour Classifier\n-----------------------------\n")
Knn(X_train, y_train, X_test, y_test)

#Test SVM before PCA
print("\n\nSupport Vector Machine Classifier\n-------------------------------\n")
Svm(X_train, y_train, X_test, y_test)

#Apply PCA to reduce dimensionality
no_c = int(input("\n------------\nApplying PCA\n------------\nNumber of components: "))

pca = PCA(n_components=no_c, svd_solver='full')
new_X_train = pca.fit_transform(X_train)
new_X_test = pca.fit_transform(X_test)

#Plot inputs
if no_c==2:
    Plot_ip(new_X_train, y_train)
    plt.title("Training data")
    #plt.show()
    Plot_ip(new_X_test, y_test, 'x')
    plt.title("Testing data")
    #plt.show()
    Plot_ip(new_X_train, y_train)
    Plot_ip(new_X_test, y_test, 'x')
    plt.legend([], [])
    plt.show()


print("\n\nAfter PCA\n---------")
#Test KNN after PCA
print("\n\nK-Nearest Neigbour Classifier\n-----------------------------\n")
neigh = Knn(new_X_train, y_train, new_X_test, y_test, 'distance')
Decision_boundary(neigh, new_X_train, y_train, 'KNN Decision Boundary')

#Test SVM before PCA
print("\n\nSupport Vector Machine Classifier\n-------------------------------\n")
clf = Svm(new_X_train, y_train, new_X_test, y_test, 'linear')
Decision_boundary(clf, new_X_train, y_train, 'SVM Decision Boundary')