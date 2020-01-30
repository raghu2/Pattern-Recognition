import numpy as np
import pandas as pd

data = pd.read_csv("iris.csv")

X = data.values
labels = X[:,-1]

X = X[:,0:3]
X1 = X[:50]
X2 = X[50:100]
X3 = X[100:150]

def getndimensionalvariance(cov):
    var = 0
    for i in range(cov.shape[0]):
        var += cov[i][i]
    return var/cov.shape[0]

def getgcase1(X, mean, cov, prior):
    wi = np.array(mean / np.square(cov))
    wi0 = -((np.matmul(np.transpose(mean.astype(float)),mean.astype(float))) / (2 * np.square(cov))) + np.log(prior)
    return (np.transpose(wi) * X + wi0)

def getgcase2(X, mean, cov, prior):
    wi = np.array(mean * np.linalg.inv(cov))
    wi0 = -(1/2) * (np.matmul(np.matmul(np.transpose(mean.astype(float)), np.linalg.inv(cov)), mean.astype(float))) + np.log(prior)
    return (np.matmul(np.transpose(wi.astype(float)), np.transpose(X.astype(float)))), wi0

def getgcase3(X, mean, cov, prior):
    Wi = -(1 / 2) * np.linalg.inv(cov)
    wi = mean * np.linalg.inv(cov)
    wi0 = -(1/2) * (np.matmul(np.matmul(np.transpose(mean.astype(float)), np.linalg.inv(cov)), mean.astype(float))) - (1 / 2) * np.log(cov) + np.log(prior)
    #print(X.shape, Wi.shape, wi.shape)
    return np.matmul(np.transpose(X.astype(float)), np.transpose(np.matmul(Wi.astype(float), np.transpose((X.astype(float)))))), np.matmul(np.transpose(wi.astype(float)), np.transpose(X.astype(float))), wi0

def case1(X1, X2, X3, prior1, prior2, prior3):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean3 = X3.mean(axis=0)
    cov1 = np.cov(np.transpose(X1.astype(float)))
    cov2 = np.cov(np.transpose(X2.astype(float)))
    cov3 = np.cov(np.transpose(X3.astype(float)))
    var1 = getndimensionalvariance(cov1)
    var2 = getndimensionalvariance(cov2)
    var3 = getndimensionalvariance(cov3)
    var = (var1 + var2 + var3) / 3
    cov = np.identity(cov1.shape[0])
    cov = var * cov
    g1 = getgcase1(X1, mean1, var, prior1)
    g2 = getgcase1(X2, mean2, var, prior2)
    g3 = getgcase1(X3, mean3, var, prior3)

def case2(X1, X2, X3, prior1, prior2, prior3):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean3 = X3.mean(axis=0)
    cov1 = np.cov(np.transpose(X1.astype(float)))
    cov2 = np.cov(np.transpose(X2.astype(float)))
    cov3 = np.cov(np.transpose(X3.astype(float)))
    cov = (cov1 + cov2 +cov3) / 3
    g1 = getgcase2(X1, mean1, cov, prior1)
    g2 = getgcase2(X2, mean2, cov, prior2)
    g3 = getgcase2(X3, mean3, cov, prior3)

def case3(X1, X2, X3, prior1, prior2, prior3):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean3 = X3.mean(axis=0)
    cov1 = np.cov(np.transpose(X1.astype(float)))
    cov2 = np.cov(np.transpose(X2.astype(float)))
    cov3 = np.cov(np.transpose(X3.astype(float)))
    #for i in range (len(X1)):
    g1 = getgcase3(X1, mean1, cov1, prior1)
    g2 = getgcase3(X2, mean2, cov2, prior2)
    g3 = getgcase3(X3, mean3, cov3, prior3)
    #return g1, g2, g3

print(case1(X1, X2, X3, 0.4, 0.3, 0.3))
print(case2(X1, X2, X3, 0.4, 0.3, 0.3))
print(case3(X1, X2, X3, 0.4, 0.3, 0.3))