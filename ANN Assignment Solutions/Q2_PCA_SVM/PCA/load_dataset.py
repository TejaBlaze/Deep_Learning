import csv
from sklearn import svm
from random import randint
from sklearn.model_selection import train_test_split
import numpy as np

def LoadData(fn):
    data, tg = [],[]
    with open(fn, 'rt') as cf:
        cr = csv.reader(cf, delimiter=' ', quotechar='|')
        for row in cr:
            sample = [ele for ele in row[1:] if ele!='']
            data.append([float(ele) for ele in sample[:-1]]), tg.append(sample[-1])            
    s = set(tg)
    s = sorted(list(s))
    dict = {}
    for i in range(len(s)):
        dict[s[i]]=i
    with open('Dataset/classmap.txt','wt') as f2:
        for key,val in dict.items():
            f2.write(str(val)+' '+str(key)+'\n')
    targets = []
    for ele in tg:
        targets.append(dict[ele])
    return [data, targets]


def check(curr_sample, samples):
    for sample in samples:
        sm = sample[0]
        flag=1
        for i,j in zip(sm,curr_sample[0]):
            if i!=j:
                flag=0
                break
        if flag==1:
            return True
    return False

def SplitData(samples, m, n):
    #print(len(samples))
    t1,t2=[],[]
    for i in range(0, len(samples)-n, n):
        prev_samples = []
        for j in range(m):
            cho = randint(i, i+n-1)
            new_sample = samples[cho]
            flag=0
            if(j==0):
                prev_samples.append(new_sample)
            else:
                if check(new_sample, prev_samples)==True:
                    flag=1
                while flag==1:
                    cho = randint(i, i+n)
                    new_sample = samples[cho]
                    flag=0
                    if check(new_sample, prev_samples)==True:
                        flag=1
                prev_samples.append(new_sample)
        t1+=[sample for sample in prev_samples]
        t2+=[sample for sample in samples[i:i+n] if check(sample,prev_samples)==False]
    return [t1,t2]

data, targets = LoadData('Dataset/ecoli.csv')
samples = []
X, y = [],[]
for xi, di in zip(data, targets):
    samples.append([xi, di])
    X.append(xi), y.append(di)

x = input("Enter ratio of sampling(m/n): ")
m,n = map(int, x.split())
#train_samples, test_samples = SplitData(samples, m, n)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-float(m/float(n)), random_state=42)
train_samples, test_samples = [],[]
for xi,di in zip(X_train, y_train):
    train_samples.append([xi, di])
for xi,di in zip(X_test, y_test):
    test_samples.append([xi, di])


with open('Dataset/ecoli_train.csv','wt') as fp:
    for ele in train_samples:
        fp.write(','.join(str(e1) for e1 in ele[0])+','+str(ele[1])+'\n')

with open('Dataset/ecoli_test.csv','wt') as fp:
    for ele in test_samples:
        fp.write(','.join(str(e1) for e1 in ele[0])+','+str(ele[1])+'\n')