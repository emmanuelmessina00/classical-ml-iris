import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from scripts.utils import *
from scripts.loglikelihood import *

def get_params_MVG(DTR,LTR):
    params=[]
    for i in np.unique(LTR):
        D0 = DTR[:,LTR==i]
        mu = getMu(D0)
        C= getC(D0)
        params.append((mu,C))
    

    return params
def getS(params,DVAL):
    res=[]
    for i in range(len(params)):
        mu=np.array(params[i][0]).reshape(DVAL.shape[0],1)
        res.append(logpdf_GAU_ND(DVAL,mu,params[i][1]))
    
    logS=np.stack(res)
    return logS

def getSJoint(params,DVAL,prior):
    return np.exp(getS(params,DVAL))*prior

def getSPost(params,DVAL,prior):
    Sjoint=getSJoint(params,DVAL,prior)
    Smarginal=getRow(Sjoint.sum(0))
    SPost=Sjoint/Smarginal
    return SPost
def get_params_Naive_Bayes(DTR,LTR):
    params=[]
    for i in np.unique(LTR):
        D0 = DTR[:,LTR==i]
        mu = getMu(D0)
        C= getC(D0) * np.eye(DTR.shape[0]) #piccola aggiunta per rendere diagonale la matrice
        params.append((mu,C))
    

    return params

def get_params_Tied_Covariance(DTR,LTR):
    params=[]
    for i in np.unique(LTR):
        D0 = DTR[:,LTR==i]
        mu = getMu(D0)
        params.append(mu)
    
    _,Sw=compute_Sb_Sw(DTR,LTR)
    params.append(Sw)
    return params
def getS_Tied_Covariance(params,DVAL):
    res=[]
    for i in range(len(params)-1):
        mu=np.array(params[i]).reshape(DVAL.shape[0],1)
        res.append(logpdf_GAU_ND(DVAL,mu,params[len(params)-1]))
    
    logS=np.stack(res)
    S=np.exp(logS)
    return S

def getSJoint_Tied_Covariance(params,DVAL,prior):
    return getS_Tied_Covariance(params,DVAL)*prior

def getSPost_Tied_Covariance(params,DVAL,prior):
    Sjoint=getSJoint_Tied_Covariance(params,DVAL,prior)
    Smarginal=getRow(Sjoint.sum(0))
    SPost=Sjoint/Smarginal
    return SPost

if __name__ == "__main__":
    
    D,L,classes,features=init_dataset()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    params=get_params_MVG(DTR,LTR)
    Spost=getSPost(params,DVAL,1/3)
    predictions=Spost.argmax(axis=0)
    accuracy=np.where(predictions == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for MVG is: {err*100}% and the accuracy: {accuracy*100}%")
    
    params=get_params_Naive_Bayes(DTR,LTR)
    Spost=getSPost(params,DVAL,1/3)
    predictions=Spost.argmax(axis=0)
    accuracy=np.where(predictions == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for Naive Bayes is: {err*100}% and the accuracy: {accuracy*100}%")
    
    _,Sw=compute_Sb_Sw(DTR,LTR)
    params_TCG=get_params_Tied_Covariance(DTR,LTR)
    Spost=getSPost_Tied_Covariance(params_TCG,DVAL,1/3)
    predictions=Spost.argmax(axis=0)
    accuracy=np.where(predictions == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for Tied Covariance Matrix is: {err*100}% and the accuracy: {accuracy*100}%")

    DB,LB=init_dataset_for_binary_classification()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(DB, LB)
    params=get_params_MVG(DTR,LTR)
    S=getS(params,DVAL)
    log_versicolor=S[0,:]
    log_virginica=S[1,:]

    s=log_virginica-log_versicolor
    t=0
    predictions_binary_classifier=np.where(s>=0,2,1)
    accuracy=np.where(predictions_binary_classifier == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for Binary Classifier is: {err*100}% and the accuracy: {accuracy*100}%")