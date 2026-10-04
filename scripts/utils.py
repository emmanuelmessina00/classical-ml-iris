import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy

def init_dataset():
    iris_dataset = load_iris()
    D = iris_dataset.data.T
    L = iris_dataset.target
    classes = iris_dataset.target_names
    features = iris_dataset.feature_names

    print(f"Le classi sono : {classes} e le features :{features}")

    return D, L, classes, features


def get_hist(D, L, features, classes):
    n_features = D.shape[0]
    for i in range(n_features):
        plt.figure()
        plt.xlabel(f"{features[i]}")
        plt.ylabel("Density")
        for c in range(len(classes)):
            mask = L == c
            D0 = D[:, mask][i, :]
            plt.hist(D0, bins=15, density=True, alpha=0.5, label=classes[c])

        plt.legend()
        plt.show()


def get_scatters(D, L, features, classes):
    n_features = D.shape[0]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            plt.figure()
            plt.xlabel(f"{features[i]}")
            plt.ylabel(f"{features[j]}")
            for c in range(len(classes)):
                mask = L == c
                D0 = D[:, mask]
                plt.scatter(D0[i, :], D0[j, :], label=classes[c])

            plt.legend()
            plt.show()

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


def compute_Sb_Sw(D, L):
    Sw = 0
    Sb = 0
    N = D.shape[1]
    mu = getCol(getMu(D))
    # Itera sulle classi uniche presenti nei dati, non su range(len(classes))
    for c in np.unique(L):
        mask = L == c
        D0 = D[:, mask]
        nc = D0.shape[1]
        mc = getCol(getMu(D0))
        Sb += nc * ((mc - mu) @ (mc - mu).T)
        DC = D0 - mc
        Sw += (DC @ DC.T)
    return Sb / N, Sw / N


def init_dataset_for_binary_classification():
    iris_dataset = load_iris()
    D_Iris = iris_dataset.data.T

    L_Iris = iris_dataset.target
    D = D_Iris[:, L_Iris != 0]
    L = L_Iris[L_Iris != 0]

    return D, L

def load_iris_binary():
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
if __name__ == "__main__":
    pass