import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from dimensionalityreduction import getC,getCol,getMu,getRow,PCA,LDA,load_iris

def init_dataset_for_binary_classification():
    iris_dataset=load_iris()
    D_Iris=iris_dataset.data.T
    
    L_Iris=iris_dataset.target
    D=D_Iris[:,L_Iris!=0]
    L=L_Iris[L_Iris!=0]

    
    return D,L

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

def LDA_classifier(DTR,LTR,DVAL,LVAL):

    # Identifica le classi uniche presenti nei dati (1 e 2, non 0-1-2)
    unique_classes = np.unique(LTR)
    
    # Controlla l'orientamento: la media di Virginica (2) deve essere > media di Versicolor (1)
    W,DTR_lda = LDA(DTR,LTR,unique_classes,1)
    mean_versicolor = DTR_lda[0, LTR==1].mean()
    mean_virginica = DTR_lda[0, LTR==2].mean()

    # Se l'orientamento è sbagliato, inverti W
    # Vogliamo che la classe Virginica (2) sia sulla destra
    if mean_virginica < mean_versicolor:
        W = -W
        DTR_lda = W.T @ DTR
        mean_versicolor = DTR_lda[0, LTR==1].mean()
        mean_virginica = DTR_lda[0, LTR==2].mean()
    
    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 #Projected samples have only 1 dimension
    DVAL_lda=W.T @ DVAL
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1

    print('Threshold:', threshold)
    print('Mean Versicolor (class 1):', mean_versicolor, 'Mean Virginica (class 2):', mean_virginica)
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

def LDA_with_PCA_classifier(DTR,LTR,DVAL,LVAL,m):
    print(f"\n--- PCA with m={m} dimensions ---")
    
    # Skip if m=0 (no PCA dimensionality)
    if m == 0:
        print("Skipping m=0 (no projection)")
        return
    
    # Applica PCA: proietta i dati di training e validation
    P, DTR_pca = PCA(DTR, getC(DTR), m)
    DVAL_pca = P.T @ DVAL
    
    # Identifica le classi uniche presenti nei dati (1 e 2, non 0-1-2)
    unique_classes = np.unique(LTR)
    
    # Applica LDA sui dati già proiettati da PCA
    W, DTR_lda = LDA(DTR_pca, LTR, unique_classes, 1)
    mean_versicolor = DTR_lda[0, LTR==1].mean()
    mean_virginica = DTR_lda[0, LTR==2].mean()

    # Se l'orientamento è sbagliato, inverti W
    # Vogliamo che la classe Virginica (2) sia sulla destra
    if mean_virginica < mean_versicolor:
        W = -W
        DTR_lda = W.T @ DTR_pca  # Riproietta i dati PCA, non i dati originali
        mean_versicolor = DTR_lda[0, LTR==1].mean()
        mean_virginica = DTR_lda[0, LTR==2].mean()
    
    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0
    DVAL_lda = W.T @ DVAL_pca  # Proietta i dati validation già trasformati da PCA
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    
    print('Threshold:', threshold)
    print('Mean Versicolor (class 1):', mean_versicolor, 'Mean Virginica (class 2):', mean_virginica)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))


if __name__=="__main__":
    DB,LB=init_dataset_for_binary_classification()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(DB, LB)
    print("\n------------------------------------------- LDA Classifier -------------------------------------------\n")
    LDA_classifier(DTR,LTR,DVAL,LVAL)
    print("\n------------------------------------------- LDA + PCA Classifier -------------------------------------------\n")
    for m in range(1, 5):  # m da 1 a 4 (escludendo 0 e 4 secondo le specifiche)
        LDA_with_PCA_classifier(DTR,LTR,DVAL,LVAL,m)