import numpy as np
import scipy.optimize

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
	# Ordinamento degli score (LLR) per ricavare s_1...s_M
	sorted_llr = np.sort(llr)

	# Creazione del vettore delle soglie includendo gli estremi -inf e +inf
	thresholds = np.concatenate([np.array([-np.inf]), sorted_llr, np.array([np.inf])])

	# Inizializzazione del minimo a un valore infinitamente grande
	min_dcf = np.inf

	# Iterazione su tutte le possibili soglie t
	for t in thresholds:
		# Calcolo delle predizioni basate sulla soglia corrente
		predictions = np.where(llr > t, 1, 0)

		# Calcolo della DCF normalizzata
		dcf_n = DCF_norm(predictions, L, pi, Cfn, Cfp)

		# Aggiornamento del valore minimo se la soglia corrente risulta più performante
		if dcf_n < min_dcf:
			min_dcf = dcf_n

	return min_dcf

if __name__ == "__main__":
    pass