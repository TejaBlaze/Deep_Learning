'''
SVD
----
Given:
Matrix A: [m,n]

Steps:
Decompose into eigen : A -> Sig * Vec
'''

import numpy as np

m,n = map(int, input("Enter (m,n) the dimensions of Array: ").split())
m,n = int(m), int(n)
arr = []
print("Enter the array A: ")
for i in range(m):
    arr.append(list(map(int, input().split())))


#Input array
A = np.array(arr)
At = A.transpose()

print("\nArray after AT * A (S):")
S = np.matmul(At, A)

print(S)

Eig_vals, V = np.linalg.eig(S)

print("\nEigen Decompositon:")
print("\nEigen Values:")
print(Eig_vals)
print("\nEigen Vectors:")
print(V)

Sigs = [ele**0.5 for ele in Eig_vals]

#Sigs = np.diag(Sigs)

U = []

for i in range(len(V)):
    ui = (1.0/Sigs[i]) * np.matmul(S, V[i])
    U.append(ui)

U = np.array(U)
print("\nU:")
print(U)

Sigs = np.diag(Sigs)
print("\nSigma: ")
print(Sigs)
print("\nV: ")
print(V)
Rhs = np.matmul(np.matmul(U, Sigs), V)
Rh = []
for i in range(len(Rhs)):
    Rh.append([int(ele) for ele in Rhs[i]])

print("Matrix at RHS = (U*Sigma*V)")
print(np.array(Rh))
