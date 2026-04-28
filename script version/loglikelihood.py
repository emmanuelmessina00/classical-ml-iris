
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from pca_lda_classifiers import getRow
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
    
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(getRow(XPlot), m, C)))
    plt.show()

    ll=loglikelihood(XPlot,m,C)
    print(f"The loglikelihood is {ll}")
    