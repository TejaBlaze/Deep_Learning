import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm

def Load_data(sample_ratio):
    iris = datasets.load_iris()
    X = iris.data[:100,:]  # we only take the first two features.
    y = iris.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-sample_ratio)
    return [X_train, X_test, y_train, y_test]

def Plot_ip(X, y):
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
        plt.scatter(xx, yy, color=c, label=class_dict[i])
    plt.title('Iris dataset in 2D')
    plt.legend(loc='best')
    plt.show()

def Knn(X_train, X_test, y_train, nn=3):
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(X_train , y_train)
    y_pred = neigh.predict(X_test)
    return y_pred
    
def Svm(X_train, X_test, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

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
    acc = float(corr/len(y_train))
    print("Confusion Matrix")
    print("-----------------")
    print("\t\t Predicted Yes\t Predicted No")
    print("\t\t -------------\t ------------")
    print("Actual Yes\t | {}\t\t | {}".format(tp, fn))
    print("Actual No\t | {}\t\t | {}".format(tn, fp))
    print("\n\nAccuracy: {:.2f}\nError rate: {:.2f}".format(acc, 1-acc))
    
#Load the dataset
x = input("Enter ratio of sampling(m/n): ")
m,n = map(int, x.split())
X_train, X_test, y_train, y_test = Load_data(float(m/n))

#Test KNN before PCA
print("\n\nK-Nearest Neigbour Classifier\n-----------------------------\n")
knn_y_pred = Knn(X_train, X_test, y_train)
Evaluate_performance(y_test, knn_y_pred)

#Test SVM before PCA
print("\n\nSupport Vector Machine Classifier\n-------------------------------\n")
svm_y_pred = Svm(X_train, X_test, y_train)
Evaluate_performance(y_test, svm_y_pred)

#Apply PCA to reduce dimensionality
no_c = int(input("PCA\nNumber of components: "))

pca = PCA(n_components=no_c, svd_solver='full')
new_X_train = pca.fit_transform(X_train)
new_X_test = pca.fit_transform(X_test)

#Plot inputs
if no_c==2:
    Plot_ip(new_X_train, y_train)