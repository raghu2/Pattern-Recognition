import numpy as np
import pandas as pd
import math

data = pd.read_csv("data_dhs_chap2.csv")
X = data.values

labels = X[:,-1]
X = X[:,0:3]
print(X)
"""
X1 = X[:10]
X2 = X[10:20]
X3 = X[20:]

#mean = X1.mean(axis=0)
cov = np.cov(np.transpose(X1))
#prior = 0.5

def mean_difference(X, mean):
    mean = mean.reshape(X.shape[1], 1)
    diff = np.zeros(shape = X.shape)
    for i in range(len(X)):
        diff[i,:] = X[i,:] - mean
    return diff

def euclidian_distance(X, mean):
    meandiff = mean_difference(X, mean)
    return np.matmul(np.transpose(meandiff, axes=(0, 2, 1)), meandiff)

def mahalanobis_distance(X, mean, cov_inv):
    meandiff = mean_difference(X, mean)
    if len(cov_inv.shape)==0:
        return meandiff * cov_inv * meandiff
    else:
        return np.matmul(np.matmul(np.transpose(meandiff, axes=(0, 2, 1)), cov_inv), meandiff)

def getDF(X, mean, cov, prior):
    cov_inv = 1/cov if len(cov.shape)==0 else np.linalg.inv(cov)
    mdist = mahalanobis_distance(X, mean, cov_inv)
    temp = -(1/2) * mdist
    d = X1.shape[1]
    lnd = -(d/2) * np.log(2 * np.pi)
    det = cov if len(cov.shape)==0 else np.linalg.det(cov)
    lnsigma = -(1/2) * np.log(det)
    lnprior = np.log(prior)
    return temp - lnd - lnsigma + lnprior

def dichotomizer(X1, X2, mean1, mean2, cov1, cov2, prior1, prior2):
    X = np.vstack((X1, X2))
    g1 = getDF(X1.reshape(X1.shape[0], X1.shape[1], -1), mean1, cov1, prior1)
    g2 = getDF(X2.reshape(X2.shape[0], X2.shape[1], -1), mean2, cov2, prior2)
    return g1-g2

def getndimensionalvariance(cov):
    var = 0
    for i in range(cov.shape[0]):
        var += cov[i][i]
    return var/cov.shape[0]

def case1(X1, X2, prior1, prior2):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    cov1 = np.cov(np.transpose(X1))
    cov2 = np.cov(np.transpose(X2))
    var1 = getndimensionalvariance(cov1)
    var2 = getndimensionalvariance(cov2)
    var = (var1 + var2)/2
    cov = np.identity(cov1.shape[0])
    cov = var * cov
    return dichotomizer(X1, X2, mean1, mean2, cov, cov, prior1, prior2)

def case2(X1, X2, prior1, prior2):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    cov1 = np.cov(np.transpose(X1))
    cov2 = np.cov(np.transpose(X2))
    cov = (cov1 + cov2)/2
    return dichotomizer(X1, X2, mean1, mean2, cov, cov, prior1, prior2)

def case3(X1, X2, prior1, prior2):
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    cov1 = np.cov(np.transpose(X1))
    cov2 = np.cov(np.transpose(X2))
    return dichotomizer(X1, X2, mean1, mean2, cov1, cov2, prior1, prior2)

print("\nConsidering only one feature x1\n")
Xi1 = X1[:,0:1]
Xi2 = X2[:,0:1]
difference = case2(Xi1, Xi2, 0.4, 0.6)
category1 = np.sum(difference[:10] > 0)
category2 = np.sum(difference[10:20] < 0)
print("Accuracy = ", (category2 + category1)/10)

print("\nConsidering two features x1, x2\n")
Xi1 = X1[:,0:2]
Xi2 = X2[:,0:2]
difference = case2(Xi1, Xi2, 0.5, 0.5)
category1 = np.sum(difference[:10] > 0)
category2 = np.sum(difference[10:20] < 0)
print("Accuracy = ", (category1+category2)/10)

print("\nConsidering three features x1, x2, x3\n")
Xi1 = X1[:,0:3]
Xi2 = X2[:,0:3]
difference = case3(Xi1, Xi2, 0.8, 0.2)
category1 = np.sum(difference[:10] > 0)
category2 = np.sum(difference[10:20] < 0)
print("Accuracy = ", (category1+category2)/10)

def threecategoryclassifier(X1, X2, X3, prior1, prior2, prior3, data):
    cov1 = np.cov(X1)
    cov2 = np.cov(X2)
    cov3 = np.cov(X3)
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean3 = X3.mean(axis=0)
    distance1 = np.log(prior1) - mahalanobis_distance(data, mean1, np.linalg.inv(cov))
    distance2 = np.log(prior2) - mahalanobis_distance(data, mean2, np.linalg.inv(cov))
    distance3 = np.log(prior3) - mahalanobis_distance(data, mean3, np.linalg.inv(cov))
    label = np.zeros(len(distance1))
    for i in range(len(distance1)):
        if distance1[i] > distance2[i] and distance1[i] > distance3[i]:
            label[i] = 0
        if distance2[i] > distance1[i] and distance2[i] > distance3[i]:
            label[i] = 1
        if distance3[i] > distance2[i] and distance3[i] > distance1[i]:
            label[i] = 2
    return label

data = np.array(((1, 2, 1), (5, 3, 2), (0, 0, 0), (1, 0, 0)))
data = data.reshape(4,3,1)
print("\nEqual probabilities\n")
print(threecategoryclassifier(X1, X2, X3, (1/3), (1/3), (1/3), data))
print("\nProbabilities 0.8, 0.1, 0.1\n")
print(threecategoryclassifier(X1, X2, X3, 0.8, 0.1, 0.1, data))"""