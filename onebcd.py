import numpy as np
import pandas as pd
import math

data = pd.read_csv("data_dhs_chap2.csv")
X = data.values

labels = X[:,-1]
X = X[:,0:3]

X1 = X[:10]
X2 = X[10:20]
X3 = X[20:]

mean = X1.mean(axis=0)
cov = np.cov(np.transpose(X1))
prior = 0.5

def mean_difference(X, mean):
    mean = mean.reshape(X.shape[1], 1)
    diff = np.zeros(shape = X.shape)
    for i in range(len(X)):
        diff[i,:] = X[i,:] - mean
    return diff

def euclidian_distance(X, mean):
    meandiff = mean_difference(X, mean)
    return np.matmul(np.transpose(meandiff), meandiff)

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

print("Disciminant Function\n ", getDF((X1.reshape(X1.shape[0], X1.shape[1], -1)), mean, cov, prior),
      "\nMahalanobis distance\n", mahalanobis_distance((X1.reshape(X1.shape[0], X1.shape[1], -1)), mean, cov))
      #"\nEuclidian distance\n", euclidian_distance(X1, mean))
