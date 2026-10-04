
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scripts.utils import getRow
import scipy

def logpdf_GAU_ND(x, mu, C):
    M=x.shape[0]
    XC = x - mu
    sign, logdet = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    quad = np.sum(XC * (invC @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + logdet + quad)

def loglikelihood(XND, m_ML, C_ML):
    logpdf=logpdf_GAU_ND(XND,m_ML,C_ML)
    return np.sum(logpdf)



if __name__=="__main__":
    pass
    