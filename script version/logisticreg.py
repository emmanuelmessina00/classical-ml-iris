import numpy as np
import scipy.optimize
import sklearn.datasets


def load_iris_binary():
	D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
	D = D[:, L != 0]  # remove setosa
	L = L[L != 0]     # remove setosa labels
	L[L == 2] = 0     # virginica -> 0, versicolor -> 1
	return D, L


def split_db_2to1(D, L, seed=0):
	nTrain = int(D.shape[1] * 2.0 / 3.0)
	np.random.seed(seed)
	idx = np.random.permutation(D.shape[1])
	idxTrain = idx[0:nTrain]
	idxTest = idx[nTrain:]
	DTR = D[:, idxTrain]
	DVAL = D[:, idxTest]
	LTR = L[idxTrain]
	LVAL = L[idxTest]
	return (DTR, LTR), (DVAL, LVAL)


def trainLogReg(DTR, LTR, l):
	ZTR = 2 * LTR - 1

	def logreg_obj(v):
		w, b = v[0:-1], v[-1]
		w = w.reshape(-1, 1)

		S = (np.dot(w.T, DTR) + b).ravel()
		loss_terms = np.logaddexp(0, -ZTR * S)
		J = (l / 2) * np.linalg.norm(w) ** 2 + np.mean(loss_terms)

		G = -ZTR / (1.0 + np.exp(ZTR * S))
		grad_w = l * w.ravel() + np.mean(G * DTR, axis=1)
		grad_b = np.mean(G)
		v_grad = np.hstack([grad_w, grad_b])

		return J, v_grad

	x0 = np.zeros(DTR.shape[0] + 1)
	xf, f_min, _ = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, approx_grad=False)
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


def get_confusion_mat(predictions, L):
	labels = np.unique(L)
	M = np.zeros((len(labels), len(labels)))

	predictions_int = np.array(predictions).astype(int)
	L_int = np.array(L).astype(int)

	for i in range(len(predictions_int)):
		M[predictions_int[i]][L_int[i]] += 1

	return M


def DCF_u(predictions, L, prior, Cfn, Cfp):
	C = np.array([[0, Cfn], [Cfp, 0]])
	M = get_confusion_mat(predictions, L)
	cols_sum = np.sum(M, axis=0)
	R = M / cols_sum
	expected_cost_per_class = np.sum(R * C, axis=0)
	prior_array = np.array([1 - prior, prior])
	DCFu = np.sum(expected_cost_per_class * prior_array)
	return DCFu


def DCF_norm(predictions, L, prior, Cfn, Cfp):
	B_dummy = min(prior * Cfn, (1 - prior) * Cfp)
	dcf_u = DCF_u(predictions, L, prior, Cfn, Cfp)
	return dcf_u / B_dummy


def get_bayes_decision(llr, pi, Cfn, Cfp):
	pi_Ht = pi
	pi_Hf = 1 - pi
	t = -np.log((pi_Ht * Cfn) / (pi_Hf * Cfp))
	predictions = np.where(llr > t, 1, 0)
	return predictions


def compute_min_dcf(llr, L, pi, Cfn, Cfp):
	sorted_llr = np.sort(llr)
	thresholds = np.concatenate([np.array([-np.inf]), sorted_llr, np.array([np.inf])])
	min_dcf = np.inf

	for t in thresholds:
		predictions = np.where(llr > t, 1, 0)
		dcf_n = DCF_norm(predictions, L, pi, Cfn, Cfp)
		if dcf_n < min_dcf:
			min_dcf = dcf_n

	return min_dcf


if __name__ == "__main__":
	D, L = load_iris_binary()
	(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

	lambdas = [1e-3, 1e-1, 1.0]
	pi_emp = np.sum(LTR == 1) / LTR.shape[0]
	emp_prior_log_odds = np.log(pi_emp / (1 - pi_emp))

	print("=== Logistic Regression (standard) ===")
	for l in lambdas:
		v_opt, J_min = trainLogReg(DTR, LTR, l)
		w_opt = v_opt[:-1]
		b_opt = v_opt[-1]

		S_val = np.dot(w_opt.T, DVAL) + b_opt
		LP = (S_val > 0).astype(int)
		error_rate = np.mean(LP != LVAL)
		llr = S_val - emp_prior_log_odds
		min_dcf = compute_min_dcf(llr, LVAL, 0.5, 1, 1)

		predictions_bayes = get_bayes_decision(llr, 0.5, 1, 1)
		act_dcf = DCF_norm(predictions_bayes, LVAL, 0.5, 1, 1)

		print(f"Lambda: {l}")
		print(f"J ottima: {J_min:e}")
		print(f"Error Rate: {error_rate * 100:.1f}%")
		print(f"DCF min: {min_dcf}")
		print(f"Actual DCF: {act_dcf}\n")

	pi_T = 0.8
	target_prior_log_odds = np.log(pi_T / (1 - pi_T))

	print("=== Logistic Regression (weighted) ===")
	for l in lambdas:
		v_opt, J_min = weighted_trainLogReg(DTR, LTR, l, pi_T)
		w_opt = v_opt[:-1]
		b_opt = v_opt[-1]

		S_val = np.dot(w_opt.T, DVAL) + b_opt
		LP = (S_val > 0).astype(int)
		error_rate = np.mean(LP != LVAL)
		llr = S_val - target_prior_log_odds
		min_dcf = compute_min_dcf(llr, LVAL, pi_T, 1, 1)

		predictions_bayes = get_bayes_decision(llr, pi_T, 1, 1)
		act_dcf = DCF_norm(predictions_bayes, LVAL, pi_T, 1, 1)

		print(f"Lambda: {l}")
		print(f"J ottima: {J_min:e}")
		print(f"Error Rate: {error_rate * 100:.1f}%")
		print(f"DCF min: {min_dcf}")
		print(f"Actual DCF: {act_dcf}\n")
