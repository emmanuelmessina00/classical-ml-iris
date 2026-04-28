import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from plots import get_hist,get_scatters,init_dataset
def getMu(D):
    return D.sum(axis=1)/D.shape[1]

def getRow(vet):
    return vet.reshape((1, vet.size))
def getCol(vet):
     return vet.reshape((vet.size, 1))

def getC(D):
    mu=getMu(D)
    mu=getCol(mu)
    Dc=D-mu
    return (Dc @ Dc.T)/Dc.shape[1]
def PCA(x,C,m):
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, x)
    return P,DP  # Restituisce P (matrice di proiezione)
def compute_Sb_Sw(D,L,classes):
    Sw=0
    Sb=0
    N=D.shape[1]
    mu=getCol(getMu(D))
    # Itera sulle classi uniche presenti nei dati, non su range(len(classes))
    for c in np.unique(L):
        mask=L==c
        D0=D[:,mask]
        nc=D0.shape[1]
        mc=getCol(getMu(D0))
        Sb+=nc*((mc-mu)@(mc-mu).T)
        DC=D0-mc
        Sw+=(DC @ DC.T)
    return Sb/N,Sw/N

def LDA(D,L,classes,m):
    Sb,Sw=compute_Sb_Sw(D,L,classes)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    DP=W.T @ D
    return W,DP
def init_dataset_for_binary_classification():
    iris_dataset=load_iris()
    D_Iris=iris_dataset.data.T
    
    L_Iris=iris_dataset.target
    D=D_Iris[:,L_Iris!=0]
    L=L_Iris[L_Iris!=0]

    
    return D,L

if __name__=="__main__":
    D,L,classes,features=init_dataset()
    mu=getMu(D)
    C=getC(D)
    print(f"La media è {mu} e la matrice di Covarianza è: {C}")
    print("\nApplicando la PCA\n")
    U,DP=PCA(D,C,2)
    get_scatters(DP,L,features,classes)
    Sb,Sw=compute_Sb_Sw(D,L,classes)
    print(f"\nLa between class covariance matrix è {Sb} \n La within class covariance matrix è{Sw}")
    print("\nApplicando la LDA")
    W,DP=LDA(D,L,classes,2)
    get_scatters(DP,L,features,classes)
    