# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:06:18 2022

@author: blanc
"""

import numpy as np

def calcMI(Z1,Z2):
    P=np.dot(Z1,np.transpose(Z2))
    PXY = P/np.sum(P)
    PXPY = np.transpose([np.sum(PXY,1)])*np.sum(PXY,0)
    ind = PXY>0
    MI=np.sum(np.multiply(PXY[ind],np.log(np.divide(PXY[ind],PXPY[ind]))));
    return MI

def calcNMI(Z1,Z2):
    NMI=(2*calcMI(Z1,Z2))/(calcMI(Z1,Z1)+calcMI(Z2,Z2));
    return NMI

def arr2mat(a,K):
    N = np.size(a)
    M = np.zeros((K*N,1))
    M[np.subtract(a+np.multiply([*range(N)],K),1)] = 1
    M = np.transpose(np.reshape(M,(N,K)))
    return M