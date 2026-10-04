import numpy as np
import scipy
from scripts.utils import *

def train_SVM(K, C, DTR, LTR):
    N = DTR.shape[1]
    ones = np.ones((1, N))
    D = np.vstack([DTR, K * ones])
    G = D.T @ D

    ZTR = LTR * 2 - 1
    H = getCol(ZTR) * getRow(ZTR) * G

    def fOpt(alpha):
        Ha = H @ alpha
        loss = 0.5 * alpha.T @ Ha - alpha.sum()
        grad = Ha - np.ones(alpha.size)
        return loss, grad

    x0 = np.zeros(N)
    alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=fOpt,
        x0=x0,
        approx_grad=False,
        bounds=[(0, C) for _ in range(N)],
        factr=np.nan,
        pgtol=1e-5
    )
    wb = D @ (alpha * ZTR)
    w = wb[:-1]
    b = wb[-1] * K
    
    dual_loss,_=fOpt(alpha)
    val=1-(ZTR*(wb.T @ D))
    primal_loss=1/2*(wb.T @ wb) + C*np.where(val>0,val,0).sum()
    print(f"Primal Loss : {primal_loss} - Dual Loss : {-dual_loss}")
    return alpha,w, b


def polygrad_d(DTR,d, DVAL, c):
    return (DTR.T @ DVAL + c) ** d


def KRB(D1, D2, gamma):
    # Calcolo delle norme al quadrato delle colonne per il primo dataset
    # Il reshape(-1, 1) assicura che il risultato sia un vettore colonna
    D1_norm2 = (D1 ** 2).sum(axis=0).reshape(-1, 1)

    # Calcolo delle norme al quadrato delle colonne per il secondo dataset
    # Il reshape(1, -1) assicura che il risultato sia un vettore riga
    D2_norm2 = (D2 ** 2).sum(axis=0).reshape(1, -1)

    # Calcolo della matrice delle distanze al quadrato sfruttando il broadcasting
    dist2 = D1_norm2 + D2_norm2 - 2 * (D1.T @ D2)

    # Applicazione dell'esponenziale scalato per gamma
    return np.exp(-gamma * dist2)


def regular_kernel(kernel, xi):
    return kernel + xi


def train_kernel_poly_SVM(K, C, DTR, LTR, xi, d, c):
    N = DTR.shape[1]
    ZTR = LTR * 2 - 1

    kernel_reg = regular_kernel(polygrad_d(DTR,d, DTR, c), xi)

    H = getCol(ZTR) * getRow(ZTR) * kernel_reg

    def fOpt(alpha):
        Ha = H @ alpha
        loss = 0.5 * alpha.T @ Ha - alpha.sum()
        grad = Ha - np.ones(alpha.size)
        return loss, grad

    x0 = np.zeros(N)
    alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=fOpt,
        x0=x0,
        approx_grad=False,
        bounds=[(0, C) for _ in range(N)],
        factr=np.nan,
        pgtol=1e-5
    )

    dual_loss, _ = fOpt(alpha)
    primal_loss = 0.5 * alpha.T @ (H @ alpha) + C * np.maximum(0, 1 - H @ alpha).sum()
    print(f"Primal Loss : {primal_loss} - Dual Loss : {-dual_loss}")
    return alpha


def train_kernel_KRB(K, C, DTR, LTR, xi, gamma):
    N = DTR.shape[1]
    ZTR = LTR * 2 - 1

    kernel_reg = regular_kernel(KRB(DTR, DTR, gamma), xi)

    H = getCol(ZTR) * getRow(ZTR) * kernel_reg

    def fOpt(alpha):
        Ha = H @ alpha
        loss = 0.5 * alpha.T @ Ha - alpha.sum()
        grad = Ha - np.ones(alpha.size)
        return loss, grad

    x0 = np.zeros(N)
    alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=fOpt,
        x0=x0,
        approx_grad=False,
        bounds=[(0, C) for _ in range(N)],
        factr=np.nan,
        pgtol=1e-5
    )

    dual_loss, _ = fOpt(alpha)
    primal_loss = 0.5 * alpha.T @ (H @ alpha) + C * np.maximum(0, 1 - H @ alpha).sum()
    print(f"Primal Loss : {primal_loss} - Dual Loss : {-dual_loss}")
    return alpha



if __name__ == "__main__":
    pass