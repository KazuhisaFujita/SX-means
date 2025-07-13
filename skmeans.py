#---------------------------------------
#Since : 2014/12/14
#Update: <2018/07/24>
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt

class SKMEANS:
    """ Implementation of SKMEANS clustering algorithm.
    """
    def __init__(self, k, X, m = None):
        self.k = k # number of clusters
        self.X = X # data points
        self.samples = np.size(X, axis=0) # number of samples

        # Initialize the cluster centers
        if m is None:
            num = np.random.permutation(self.samples)
            self.m = self.X[num[0:self.k]]
        else:
            self.m = m

        self.class_num = np.zeros(self.samples) # cluster number for each sample

    def fit(self):
        QTH = 0.01 # threshold for convergence
        ENDMAX = 200 # maximum number of iterations

        oQ = float("inf")
        for t in range(ENDMAX):
            for i in range(self.samples):
                # calculate the cosine similarity between sample i and each cluster center
                # the norms of the cluster center and the sample are one.
                self.class_num[i] = np.argmax(np.sum(self.m * self.X[i], axis=1))

            for i in range(self.k):
                sum_x = np.sum(self.X[self.class_num == i], axis=0)
                self.m[i] = sum_x/np.linalg.norm(sum_x)

            nQ = 0
            for i in range(self.k):
                nQ += np.sum(self.X[self.class_num == i] * self.m[i])

            if mt.fabs(nQ - oQ) < QTH:
                break
            oQ = nQ
