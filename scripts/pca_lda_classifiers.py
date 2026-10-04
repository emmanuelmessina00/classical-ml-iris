import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from scripts.dimensionalityreduction import train_LDA,train_PCA,apply_LDA,apply_PCA

def LDA_classifier(DTR,LTR,DVAL,LVAL):

    # Identifica le classi uniche presenti nei dati (1 e 2, non 0-1-2)
    unique_classes = np.unique(LTR)
    
    # Controlla l'orientamento: la media di Virginica (2) deve essere > media di Versicolor (1)
    W=train_LDA(DTR,LTR,1)
    DTR_lda = apply_LDA(DTR,W)

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


def LDA_with_PCA_classifier(DTR, LTR, DVAL, LVAL, m):
    print(f"\n--- PCA with m={m} dimensions ---")

    if m == 0:
        print("Skipping m=0 (no projection)")
        return

    # Addestra e applica PCA (centrando anche DVAL con mu del training)
    P, mu = train_PCA(DTR, m)
    DTR_pca = apply_PCA(DTR, P, mu)
    DVAL_pca = apply_PCA(DVAL, P, mu)

    # Addestra LDA a 1 dimensione sui dati proiettati da PCA
    W = train_LDA(DTR_pca, LTR, 1)
    DTR_lda = apply_LDA(DTR_pca, W)

    mean_versicolor = DTR_lda[0, LTR == 1].mean()
    mean_virginica = DTR_lda[0, LTR == 2].mean()

    # Se l'orientamento è invertito, corregge W e riproietta
    if mean_virginica < mean_versicolor:
        W = -W
        DTR_lda = apply_LDA(DTR_pca, W)
        mean_versicolor = DTR_lda[0, LTR == 1].mean()
        mean_virginica = DTR_lda[0, LTR == 2].mean()

    # Calcolo soglia e predizione su DVAL
    threshold = (mean_versicolor + mean_virginica) / 2.0
    DVAL_lda = apply_LDA(DVAL_pca, W)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1

    print('Threshold:', threshold)
    print('Mean Versicolor (class 1):', mean_versicolor, 'Mean Virginica (class 2):', mean_virginica)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))


if __name__=="__main__":
    pass