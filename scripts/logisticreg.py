import numpy as np
import scipy.optimize
import sklearn.datasets

from scripts.utils import init_dataset,split_db_2to1,init_dataset_for_binary_classification


def trainLogReg(DTR, LTR, l):
	ZTR = 2 * LTR - 1  # definizione di z
	n = DTR.shape[1]  # numero di campioni

	def logreg_obj(v):
		w, b = v[0:-1], v[-1]
		w = w.reshape(-1, 1)

		S = (np.dot(w.T, DTR) + b).ravel()
		loss_terms = np.logaddexp(0, -ZTR * S)
		J = (l / 2) * np.linalg.norm(w) ** 2 + np.mean(loss_terms)
		G = -ZTR / (1.0 + np.exp(ZTR * S))
		# gradienti
		grad_w = l * w.ravel() + np.mean(G * DTR, axis=1)
		grad_b = np.mean(G)
		v_grad = np.hstack([grad_w, grad_b])

		return J, v_grad

	x0 = np.zeros(DTR.shape[0] + 1)
	xf, f_min, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, approx_grad=False)

	return xf, f_min


def weighted_trainLogReg(DTR, LTR, l, piT):
	ZTR = 2 * LTR - 1

	def weighed_logreg_obj(v):
		w, b = v[0:-1], v[-1]
		w = w.reshape(-1, 1)

		S = (np.dot(w.T, DTR) + b).ravel()
		nT = (LTR == 1).sum()
		nF = (LTR == 0).sum()

		xi = np.where(ZTR == 1, piT / nT, (1 - piT) / nF)
		weighed_loss_terms = xi * np.logaddexp(0, -ZTR * S)
		J = (l / 2) * np.linalg.norm(w) ** 2 + np.sum(weighed_loss_terms)

		G = -ZTR / (1.0 + np.exp(ZTR * S))
		grad_w = l * w.ravel() + np.sum(xi * G * DTR, axis=1)
		grad_b = np.sum(xi * G)
		v_grad = np.hstack([grad_w, grad_b])

		return J, v_grad

	x0 = np.zeros(DTR.shape[0] + 1)
	xf, f_min, _ = scipy.optimize.fmin_l_bfgs_b(func=weighed_logreg_obj, x0=x0, approx_grad=False)
	return xf, f_min


if __name__ == "__main__":
	pass