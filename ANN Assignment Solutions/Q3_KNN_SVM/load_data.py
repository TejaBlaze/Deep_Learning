from sklearn.model_selection import train_test_split
from sklearn import datasets

def Load_data(no_c, sample_ratio):
    if no_c==2:
        ul=100
    elif no_c==3:
        ul=150
    iris = datasets.load_iris()
    X = iris.data[:ul,:]  # we only take the first two features.
    y = iris.target[:ul]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-sample_ratio)
    return [X_train, X_test, y_train, y_test]

#Load the dataset
no_c = int(input("Enter number of classes: "))
x = input("Enter ratio of sampling(m/n): ")
m,n = map(int, x.split())
X_train, X_test, y_train, y_test = Load_data(no_c, float(m/n))
train_samples, test_samples = [],[]
for xi,di in zip(X_train, y_train):
    train_samples.append([xi, di])
for xi,di in zip(X_test, y_test):
    test_samples.append([xi, di])
with open('Dataset/iris_train.csv','wt') as fp:
    for ele in train_samples:
        fp.write(','.join(str(e1) for e1 in ele[0])+','+str(ele[1])+'\n')
with open('Dataset/iris_test.csv','wt') as fp:
    for ele in test_samples:
        fp.write(','.join(str(e1) for e1 in ele[0])+','+str(ele[1])+'\n')
