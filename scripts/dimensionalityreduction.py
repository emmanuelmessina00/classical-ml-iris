import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from scripts.utils import compute_Sb_Sw

def train_PCA(X, m):
    mu = np.mean(X, axis=1, keepdims=True)  # Centering Data
    X_C = X - mu  # Calculate C

    C = (X_C @ X_C.T) / X_C.shape[1]  # Eigenvectors of C

    _, eig = np.linalg.eigh(C)  # Take the m eig to build P

    P = eig[:, ::-1][:, :m]  # return P
    return P, mu
def apply_PCA(X, P, mu):
    X_C = X - mu
    return P.T @ X_C

def train_LDA(X,L,m):
    Sb,Sw=compute_Sb_Sw(X,L)
    _,U=scipy.linalg.eigh(Sb,Sw)
    W=U[:,::-1][:,:m]
    return W
def apply_LDA(X,W):
    return W.T @ X


if __name__=="__main__":
    pass