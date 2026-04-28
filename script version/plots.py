import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def init_dataset():
    iris_dataset=load_iris()
    D=iris_dataset.data.T
    L=iris_dataset.target
    classes=iris_dataset.target_names
    features=iris_dataset.feature_names
    
    print(f"Le classi sono : {classes} e le features :{features}")
    
    return D,L,classes,features


def get_hist(D,L,features,classes):
    n_features = D.shape[0]
    for i in range(n_features):
        plt.figure()
        plt.xlabel(f"{features[i]}")
        plt.ylabel("Density")
        for c in range(len(classes)):
            mask= L==c
            D0=D[:,mask][i,:]
            plt.hist(D0,bins=15,density=True,alpha=0.5,label=classes[c])
        
        
        plt.legend()
        plt.show()

def get_scatters(D,L,features,classes):
    n_features = D.shape[0]
    for i in range(n_features):
        for j in range(i+1,n_features):
            plt.figure()
            plt.xlabel(f"{features[i]}")
            plt.ylabel(f"{features[j]}")
            for c in range(len(classes)):
                mask=L==c
                D0=D[:,mask]
                plt.scatter(D0[i,:],D0[j,:],label=classes[c])
            
            plt.legend()
            plt.show()
if __name__=="__main__":
    D,L,classes,features=init_dataset()
    print("\nGenerazione histogrammi...")
    get_hist(D,L,features,classes)
    print("\nGenerazione scatters...")
    get_scatters(D,L,features,classes)
