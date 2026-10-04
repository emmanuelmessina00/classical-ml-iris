import numpy as np
import scipy
import sklearn
from scripts.gaussianclass import *


def logpdf_GMM(X,gmm):
    n_cluster=len(gmm)
    S=np.zeros((n_cluster,X.shape[1]))
    
    for g in range(n_cluster):
        w=gmm[g][0]
        mu=gmm[g][1]
        C=gmm[g][2]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    
    logdens = scipy.special.logsumexp(S, axis=0)
    return S,logdens
def EM_constrained(threshold,X,gmm,psi):
    
    actual_gmm=gmm
    N=X.shape[1]
    
    #E Step
    # Calcolo la probabilità a posteriori (responsability) con logpdf_GMM(X,gmm)
    S, logdens = logpdf_GMM(X, actual_gmm)
    ll_actual = np.mean(logdens) #Ci servirà dopo per le stop criterion
    
    while True:
    # Calcolo la probabilità a posteriori (responsability) con logpdf_GMM(X,gmm)
        gamma = np.exp(S - logdens)
        new_gmm=[]

        for g in range(len(actual_gmm)):
            gamma_g=gamma[g,:]

            Zg=np.sum(gamma_g)
            # Fg diventa un vettore colonna D x 1 grazie a keepdims=True
            Fg=np.sum(X*gamma_g,axis=1,keepdims=True)
            # Sg è la matrice D x D ottenuta dal prodotto della matrice pesata per la sua trasposta
            Sg=(X*gamma_g) @ X.T

            #Aggiorno i parametri
            new_mu = Fg / Zg
            new_cov = Sg / Zg - (new_mu @ new_mu.T)
            new_w = Zg / N

            #Applichiamo il vincolo
            U, s, _ = np.linalg.svd(new_cov)
            s[s<psi] = psi
            new_cov = np.dot(U, getCol(s)*U.T)
            
            new_gmm.append((new_w, new_mu, new_cov))
        
        #Controllo gli stop criterion
        S_new,new_logdens=logpdf_GMM(X,new_gmm)
        ll_new=np.mean(new_logdens)
        delta=ll_new-ll_actual

        if delta<=threshold:
            actual_gmm=new_gmm
            break

        actual_gmm=new_gmm
        S=S_new
        logdens=new_logdens
        ll_actual=ll_new
    return actual_gmm

def split_gmm(gmm,alpha=0.1):
    new_gmm=[]

    for w,mu,cov in gmm:
        new_w=w/2
        U, s, Vh = np.linalg.svd(cov)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        new_mu1=mu-d
        new_mu2=mu+d
        new_gmm.append((new_w,new_mu1,cov))
        new_gmm.append((new_w,new_mu2,cov))

    return new_gmm

def LBG_constrained(X,threshold,target_components,alpha=0.1,psi=0.01):
    mu_global=np.mean(X,axis=1,keepdims=True)
    cov_global = ((X - mu_global) @ (X - mu_global).T) / X.shape[1]
    GMM_1 = [(1.0, mu_global, cov_global)]

    gmm=EM_constrained(threshold,X,GMM_1,psi) #serve a stabilizzare la loglikelihood

    current_components=1

    while current_components<target_components:
        #sdoppiamento
        gmm=split_gmm(gmm,alpha)
        gmm=EM_constrained(threshold,X,gmm,psi)
        current_components*=2
    
    return gmm

if __name__ == "__main__":
    pass