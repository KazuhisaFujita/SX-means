import numpy as np
import math as mt
from skmeans import SKMEANS

class SXMEANS:
    """ Implementation of SXMEANS clustering algorithm.
    """

    def Kappa(self, r, d):
        """ Calculate the kappa parameter for von Mises-Fisher distribution.
        """
        return r*(d - r*r)/(1 - r*r)

    def __init__(self, X, fixed_kappa = 10, fixed=False):
        self.X = X
        self.k = 2
        self.samples = np.size(self.X, axis = 0)
        self.DIM = np.size(self.X, axis = 1)
        num = np.random.permutation(self.samples)
        self.m = self.X[num[0:self.k]]
        self.class_num = np.zeros(self.samples)

        self.fixed = fixed
        self.fixed_kappa = fixed_kappa

    def sxmeans(self):
        KMAX = 600 # maximum kappa

        ok = self.k
        while(1):
            #Improve Parameter
            sk = SKMEANS(self.k, self.X, self.m)
            sk.fit()
            self.m = sk.m
            self.class_num = sk.class_num

            # Calculate BIC
            p = self.DIM + 1 #the number of parameters

            obic = np.zeros(self.k)
            nbic = np.zeros(self.k)

            for i in range(self.k):
                ll = 0
                mask = (self.class_num == i)
                cx = self.X[mask]
                rn = np.size(cx, axis=0)
                rbar =  np.linalg.norm(np.sum(cx, axis=0))/float(rn)

                if self.fixed == False:
                    kappa = self.Kappa(rbar, self.DIM)
                else:
                    kappa = self.fixed_kappa

                # clipping likelihood
                # if kappa > KMAX:
                #     ll = float("inf")
                # else:
                #     ll = rn * ((self.DIM - 1) / 2.0 * mt.log(kappa) -  (self.DIM - 1) / 2.0 * mt.log(mt.pi*2) - kappa) + kappa * np.sum(np.dot(cx, self.m[i]))

                ll = rn * ((self.DIM - 1) / 2.0 * mt.log(kappa) -  (self.DIM - 1) / 2.0 * mt.log(mt.pi*2) - kappa) + kappa * np.sum(np.dot(cx, self.m[i]))
                obic[i] = ll - p/2.0*mt.log(rn)

            #Improve Structure
            u = np.zeros((self.k,2,self.DIM))
            delcl=[]

            # splitting
            for i in range(self.k):
                mask = (self.class_num == i)
                ci = self.X[mask]
                ci_size = np.size(ci,axis=0)
                ci_class_num = np.zeros(ci_size)

                # initialize parame
                # sk-means
                sk2 = SKMEANS(2, ci)
                sk2.fit()
                u[i] = sk2.m
                ci_class_num = sk2.class_num

                rn = np.size(ci,axis=0)
                rns = np.zeros(2)
                kappas = np.zeros(2)
                for j in range(2):
                    mask2 = (ci_class_num == j)
                    rns[j] = np.size(ci[mask2],axis=0)
                    rbar =  np.linalg.norm(np.sum(ci[mask2], axis=0))/float(rns[j])

                    if self.fixed == False:
                        kappas[j] = self.Kappa(rbar, self.DIM)
                    else:
                        kappas[j] = self.fixed_kappa

                # calculate log likelihood
                ll = 0

                # clip likelihood
                # if kappas[0] < KMAX and kappas[1] < KMAX and rns[0] > 0 and rns[1] > 0:
                #     for j in range(2):
                #         mask2 = (ci_class_num == j)
                #         ll += rns[j] * (mt.log(rns[j]) - mt.log(rn) - (self.DIM - 1) / 2.0 * mt.log(2*mt.pi) + (self.DIM-1) / 2.0 * mt.log(kappas[j]) - kappas[j]) + kappas[j] * np.sum(np.dot(ci[mask2], u[i][j]))
                # else:
                #     ll = float("-inf")

                for j in range(2):
                    mask2 = (ci_class_num == j)
                    ll += rns[j] * (mt.log(rns[j]) - mt.log(rn) - (self.DIM - 1) / 2.0 * mt.log(2*mt.pi) + (self.DIM-1) / 2.0 * mt.log(kappas[j]) - kappas[j]) + kappas[j] * np.sum(np.dot(ci[mask2], u[i][j]))

                # calculate BIC
                p = 2*(self.DIM + 1)
                nbic[i] = ll - p/2.0*mt.log(rn)

                # check if splitting is better
                if obic[i] < nbic[i]:
                    delcl.append(i)

            # if no cluster is split, then stop
            if len(delcl) != 0:
                for i in delcl:
                    self.m = np.append(self.m, [u[i][0]], axis=0)
                    self.m = np.append(self.m, [u[i][1]], axis=0)
                self.m = np.delete(self.m, delcl, axis=0)


            # print(obic)
            # print(nbic)
            self.k += len(delcl)

            if ok == self.k or self.k > 20:
                break
            ok = self.k

        sk = SKMEANS(self.k, self.X, self.m)
        sk.fit()
        self.class_num = sk.class_num
