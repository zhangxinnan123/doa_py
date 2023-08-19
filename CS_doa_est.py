#coding: utf-8
#k_max: maximum target number
#k_cur: current target number
#Y: array sample matrix
#G: gridsize
#L: array length
#Yt: residual
import numpy as np

def f_steer_vec(angles, L):
	l = np.array([range(L)]).T
	k = np.size(angles)
	return np.exp(-1j*np.pi*l.dot(np.sin(angles.reshape((1, k)))))

def func_fft_max(Yt, fft_num, L):
	Y_fft = np.fft.fftshift(np.fft.fft(Yt, fft_num, 0)/L, 0)
	# Y_fft = np.fft.fft(Yt.conj(), fft_num, 0)/L
	# Y_fft = np.append(Y_fft[fft_num//2:, :], Y_fft[:fft_num//2, :], axis = 0)
	Y_fft_power = np.mean(np.abs(Y_fft)**2, 1)
	index = np.argmax(Y_fft_power)
	angle = np.linspace(-1, 1, fft_num, endpoint = False)
	angle = -np.arcsin(angle)
	theta = angle[index]
	s_hat = Y_fft[index, :]
	return theta, s_hat	

def func_finesearch(Y, fft_num, L, theta_hat):
	depth = 15
	theta_hat = np.sin(theta_hat)
	
	def f_angle(x, y):
		return np.array([x - 1/fft_num/2**y, x, x + 1/fft_num/2**y])
		
	Y_fft = f_steer_vec(theta_hat, L).conj().T.dot(Y)
	max_v = np.sum(np.abs(Y_fft)**2, 1)
	
	step = 0
	while step < depth:
		step += 1
		angle = f_angle(theta_hat, step - 1)
		angle_t = np.array([angle[0], angle[2]])
		A = f_steer_vec(angle_t, L)
		Y_fft = A.conj().T.dot(Y)
		Y_fft_power = np.sum(np.abs(Y_fft)**2, 1)
		
		Y_fft_power = np.array([Y_fft_power[0], max_v, Y_fft_power[1]], dtype='float64')
		max_v = np.max(Y_fft_power)
		index = np.argmax(Y_fft_power)
		
		theta_hat = angle[index]
	theta = np.arcsin(theta_hat)
	s = f_steer_vec(theta_hat, L).conj().T.dot(Y)/L
	return theta, s
	
def func_find_support(S_hat, Sparse_rate, T_support):
	S_power = np.sum(S_hat*(S_hat.conj()), 1)
	index = np.argsort(-S_power)
	support = T_support[index[:Sparse_rate]]
	S_hat = S_hat[index[:Sparse_rate], :].copy()
	return support, S_hat
	
def func_fft_max_support(Yt, fft_num, Sparse_rate, alpha):
	Y_fft = np.fft.fft(Yt, fft_num, 0)
	Y_fft_power = np.sum(np.abs(Y_fft)**2, 1)
	power = np.append(Y_fft_power[fft_num//2:], Y_fft_power[:fft_num//2])
	index = np.argsort(-power)
	return index[:alpha*Sparse_rate]

def func_max_sparse(Y, K):
	Y_power = np.sum(Y*Y.conj(), 1)
	index = np.argsort(-Y_power)
	support = index[:K]
	Y[index[K:]] = 0
	S_hat = Y
	return S_hat, support
	
def func_BIC(Y, theta, S, source_num):
	L, T = np.shape(Y)
	l = np.array([range(L)]).T
	A = f_steer_vec(theta[:source_num], L)
	bic_part1 = 2*L*T*np.log(np.linalg.norm(Y - A.dot(S[:source_num, :])**2/L/T))
	bic_part2 = (3*source_num + 2*source_num*T)*np.log(L) + (2*source_num*T + source_num)*np.log(T)
	return  bic_part1 + bic_part2
	
def func_newton_search(Yt, theta):
	L, T = np.shape(Yt)
	
	def f_steer_vec_deriv1(angles, L):
		k = np.size(angles)
		l = np.array([range(L)]).T
		angles = angles.reshape((1, k))
		return (-1j*(np.pi)*l.dot(np.cos(angles)))*np.exp(-1j*(np.pi)*l.dot(np.sin(angles)))
		
	def f_steer_vec_deriv2(angles, L):
		l = np.array([range(L)]).T
		k = np.size(angles)
		angles = angles.reshape((1, k))
		return (1j*(np.pi)*l.dot(np.sin(angles)) - (np.pi)**2*(l**2).dot(np.cos(angles)**2))*np.exp(-1j*(np.pi)*l.dot(np.sin(angles)))
	dif = 1
	epsilon = 1e-10
	theta_pre = theta.copy()
	iter_num = 1
	while dif > epsilon:
		d = np.real((f_steer_vec_deriv1(theta_pre, L).conj().T.dot(Yt)).dot(Yt.conj().T.dot(f_steer_vec(theta_pre, L))))
		D = np.real((f_steer_vec_deriv2(theta_pre, L).conj().T.dot(Yt)).dot(Yt.conj().T.dot(f_steer_vec(theta_pre, L))) + np.sum(np.abs(f_steer_vec_deriv1(theta_pre, L).conj().T.dot(Yt))**2))
		theta_now = theta_pre - np.diag(d/D)
		if iter_num == 20: break
		iter_num += 1
		dif = np.abs((theta_now - theta_pre)/theta_pre)
		theta_pre = theta_now
	theta_hat = theta_pre
	
	S_hat = f_steer_vec(theta_hat, L).conj().T.dot(Yt)/L
	return theta_hat, S_hat

# MP
def func_MP(Y, G, k_max = 10):
	L, T = np.shape(Y)
	fft_num = G

	theta_hat = np.zeros((k_max, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))

	k_cur = 0

	Yt = np.copy(Y)

	while k_cur < k_max:
		k_cur += 1
		if k_cur > 1:
			steer_vec_cur = f_steer_vec(theta_hat[k_cur - 2], L)
			s_hat_cur = S_hat[k_cur - 2, :].reshape((1, T))
			Yt -= steer_vec_cur.dot(s_hat_cur)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
	angles = theta_hat/np.pi*180
	powers = np.mean(abs(S_hat)**2, 1)
	return angles, powers, S_hat
	

# OMP
def func_OMP(Y, G, max_source_num = 10):
	L, T = np.shape(Y)
	fft_num = G
	
	theta_hat = np.zeros((max_source_num, 1))
	S_hat = np.zeros((max_source_num, T)) + 1j*np.zeros((max_source_num, T))

	source_num_hat = 0

	Yt = np.copy(Y)

	while source_num_hat < max_source_num:
		source_num_hat += 1
		theta_hat[source_num_hat-1], S_hat[source_num_hat-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		A = f_steer_vec(theta_hat[:source_num_hat], L)
		Yt = Y - A.dot(np.linalg.inv(A.conj().T.dot(A))).dot(A.conj().T).dot(Y)
	angles = theta_hat/np.pi*180
	powers = np.mean(abs(S_hat)**2, 1)
	return angles, powers, S_hat

# CoSaMP
def func_CoSaMP(Y, G, K):
	Sparse_rate = K
	L, T = np.shape(Y)
	fft_num = G
	
	angle_bar = np.linspace(-1, 1, fft_num, endpoint = False)
	angle_bar = -np.arcsin(angle_bar).reshape((fft_num, 1))
	
	iter_num = 0
	iter_max = 20
	
	theta_hat = np.zeros((Sparse_rate, 1))
	S_hat = np.zeros((Sparse_rate, T)) + 1j*np.zeros((Sparse_rate, T))
	
	alpha = 2
	
	S_support = np.array([], dtype='int')
	Yt = Y.copy()
	while iter_num < iter_max:
		iter_num += 1
		theta_support = func_fft_max_support(Yt, fft_num, Sparse_rate, alpha)
		T_support = np.union1d(S_support, theta_support)
		A = f_steer_vec(angle_bar[T_support], L)
		S_hat = np.linalg.inv(A.conj().T.dot(A)).dot(A.conj().T).dot(Y)
		
		S_support, S_hat = func_find_support(S_hat, Sparse_rate, T_support)
		A = f_steer_vec(angle_bar[S_support], L)
		Yt = Y - A.dot(S_hat)
	
	angle = angle_bar[S_support]/np.pi*180
	power = np.mean(np.abs(S_hat)**2, 1)
	return angle, power, S_hat
		
		
# CLEAN
def func_CLEAN(Y, G, k_max = 10):
	L, T = np.shape(Y)
	fft_num = G
	
	theta_hat = np.zeros((k_max, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))

	k_cur = 0

	Yt = np.copy(Y)
	while k_cur< k_max:
		k_cur += 1
		if k_cur > 1:
			steer_vec_cur = f_steer_vec(theta_hat[k_cur - 2], L)
			s_hat_cur = S_hat[k_cur - 2, :].reshape((1, T))
			Yt -= steer_vec_cur.dot(s_hat_cur)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		# refine estimates
		# theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_finesearch(Yt.copy(), fft_num, L, theta_hat[k_cur-1])
		
		# refine estimates Newton search(CLEAN3)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Yt, theta_hat[k_cur-1])
	angles = theta_hat/np.pi*180
	powers = np.mean(abs(S_hat)**2, 1)
	return angles, powers, S_hat


# RELAX
def func_RELAX(Y, fft_num = 1024, k_max = 5, iter_max = 50, epsilon=1e-3):
	L, T = np.shape(Y)
	
	theta_hat = np.zeros((k_max, 1))
	BIC_v = np.zeros((k_max+1, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))
	
	k_cur = 1
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Y.copy(), fft_num, L)
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Y.copy(), theta_hat[k_cur-1])
	Y1 = f_steer_vec(theta_hat[k_cur - 1], L).dot(S_hat[k_cur - 1, :].reshape((1, T)))
	loglike_pre = np.linalg.norm(Y - Y1)**2
	BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur-1].copy(), S_hat[0:k_cur-1, :].copy(), k_cur)
	while k_cur < k_max:
		k_cur += 1
		Yt = np.copy(Y)
		for i in range(k_cur - 1):
			Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Yt.copy(), theta_hat[k_cur-1])
		dif = 1
		num = 1
		iter_num = 1
		while dif > epsilon:
			Yt = Y.copy()
			for i in range(k_cur):
				if i != num - 1:
					Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
# 			theta_hat[num-1], S_hat[num-1, :] = func_fft_max(Yt.copy(), fft_num, L)
			theta_hat[num-1], S_hat[num-1, :] = func_newton_search(Yt.copy(), theta_hat[num-1])
			if num == k_cur:
				if iter_num == iter_max: break
				loglike_v = np.linalg.norm(Yt - f_steer_vec(theta_hat[num-1], L).dot(S_hat[num-1, :].reshape((1, T))))**2
				dif = np.abs(loglike_v - loglike_pre)/ np.abs(loglike_pre)
				loglike_pre = loglike_v.copy()
				iter_num += 1
			num = np.mod(num, k_cur) + 1
		BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur].copy(), S_hat[0:k_cur, :].copy(), k_cur)
	theta_es = theta_hat[:k_cur]
	S_es = S_hat[:k_cur, :]
	source_num_es = k_cur
	
	angle = theta_es/np.pi*180
	power = np.mean(np.abs(S_es)**2, 1)
	
	return angle, power, source_num_es, S_es, BIC_v.reshape((k_max+1,))

# IHT	
def func_IHT(Y, G, K):
	L, T = np.shape(Y)
	fft_num = G
	
	angle = np.linspace(-1, 1, G, endpoint = False)
	angle = -np.arcsin(angle).reshape((G, 1))
	A_bar = f_steer_vec(angle, L)
	
	v, U = np.linalg.eig(A_bar.conj().T.dot(A_bar))
	tau = 1/np.max(v)
	
	S_hat = np.zeros((G, T)) + 1j*np.zeros((G, T))
	
	iter_cur = 0
	iter_max = 30
	support = np.array([], dtype = 'int')
	
	while iter_cur < iter_max:
		iter_cur += 1
		R = Y - A_bar[:, support].dot(S_hat[support, :])
		Y_fft = np.fft.fft(R, fft_num, 0)
		Y_fft = np.append(Y_fft[fft_num//2:, :], Y_fft[:fft_num//2, :], axis = 0)
		
		S_hat, support = func_max_sparse(S_hat + tau*Y_fft, K)
		
	angle = angle/np.pi*180
	power = np.mean(np.abs(S_hat)**2, 1)
	return angle, power
	
	
def find_peak(p_hat, K_max):
	K_max = 2
	G = len(p_hat)
	p_peak = np.array([0]*G)
	for k in range(1, G - 1):
		if p_hat[k] > p_hat[k + 1] and p_hat[k] > p_hat[k - 1]:
			p_peak[k] = 1
	p_idx_hat = (p_hat.reshape(G, 1)*p_peak.reshape(G, 1)).reshape(G, )
	idx_sort = sorted(range(len(p_hat)), key=lambda k: p_idx_hat[k], reverse=True)
	if len(idx_sort) <= K_max: return sorted(idx_sort)
	idx_hat = sorted(idx_sort[:K_max])

	return np.array(idx_hat), p_hat[idx_hat]
    
def func_compare_err(angle_es, angle_true, power_es, Rayleigh_res):
	if_res = 0
	K = np.size(angle_true)
	
	locs, pks = find_peak(power_es, K)
	
	sort_index = np.argsort(-pks)
	doa_es = angle_es[locs[sort_index[:K]]]
	power_t = power_es[locs[sort_index[:K]]]
	
	doa_sort = np.sort(doa_es)
	ind1 = np.argsort(doa_es)
	angle_sort = np.sort(angle_true)
	ind2 = np.argsort(angle_true)
	
	dif_t = Rayleigh_res / 2
	doa_err = np.abs(doa_sort - angle_sort)
	dist = doa_err - dif_t

	if len([k for k in dist.reshape((K,)) if k <= 0]) == K:
		if_res = 1
	else: doa_err = np.zeros(np.shape(doa_err))
	return if_res, doa_err

def func_compare_err1(angle_es, angle_true, Rayleigh_res):
	K = np.size(angle_true)
	angles_est = np.sort(angle_es)
	angles_true = np.sort(angle_true)
	doa_err = np.abs(angles_est - angles_true)
	dif_t = Rayleigh_res / 2
	dist = doa_err - dif_t
	if len([k for k in dist if k <= 0]) == K:
		return 1, doa_err.reshape((K,))      
	return 0, np.array([0]*K)


def func_iaa_relax(X, A_bar, iaa_max_iter, target_num):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = (A_bar.conj().T).dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = np.sum(s_hat*s_hat.conj(), 1).reshape((G_tot, 1))/T
	for i in range(iaa_max_iter):
		R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T) + 1e-8*np.eye(L)
		R_invA = np.linalg.inv(R_hat).dot(A_bar)
		part1 = R_invA.conj().T.dot(X)
		part2 = np.sum(R_invA.conj()*A_bar, 0).reshape((G_tot, 1))
		s_hat = part1/part2
		p_hat = np.sum(s_hat*s_hat.conj(), 1)/T
	s_hat = s_hat[:G, :]
	power = np.mean(abs(s_hat)**2, 1)
	
	index = find_peak(power, target_num)[0]	
	theta_grid = np.arcsin(np.linspace(-1, 1, G, endpoint = False))
	
	theta_hat = theta_grid[index]
	S_hat = s_hat[index, :]
	
	epsilon = 1
	num = 1
	loglike_pre = 0
	while epsilon > 1e-3:
		Yt = X.copy()
		for i in range(target_num):
			if i != num - 1:
				Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
		theta_hat[num-1], S_hat[num-1, :] = func_fft_max(Yt.copy(), 512, L)
		theta_hat[num-1], S_hat[num-1, :] = func_newton_search(Yt.copy(), theta_hat[num-1])
		if num == target_num:
			loglike_v = np.log(np.linalg.norm(Yt - f_steer_vec(theta_hat[num-1], L).dot(S_hat[i, :].reshape((1, T))))**2)
			epsilon = np.abs(loglike_v - loglike_pre)/ np.abs(loglike_v)
			loglike_pre = loglike_v.copy()
		num = np.mod(num, target_num) + 1
		
	angle = theta_hat/np.pi*180
	power = np.mean(np.abs(S_hat)**2, 1)
	
	return angle, power


def func_RELAX_BIC(Y, fft_num = 256, source_num_max = 0, max_iter = 50, epsilon = 1e-3):
	L, T = np.shape(Y)
	
	if source_num_max == 0:
		k_max = min(10, int(np.floor(2/3*L)))
	else: 
		k_max = source_num_max
	
	BIC = np.zeros(k_max)
	theta_hat = np.zeros((k_max, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))
	theta_hat_temp = np.zeros((k_max, k_max))
	S_hat_temp = np.zeros((k_max, T, k_max)) + 1j*np.zeros((k_max, T, k_max))
	
	k_cur = 1
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Y.copy(), fft_num, L)
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Y.copy(), theta_hat[k_cur-1])
	Y1 = f_steer_vec(theta_hat[k_cur - 1], L).dot(S_hat[k_cur - 1, :].reshape((1, T)))
	likelihood_pre = np.linalg.norm(Y - Y1)**2
	
	theta_hat_temp[:k_cur, k_cur - 1] = theta_hat[:k_cur]
	S_hat_temp[:k_cur, :, k_cur - 1] = S_hat[:k_cur, :]
	
	Y1 = f_steer_vec(theta_hat[:k_cur], L).dot(S_hat[:k_cur, :].reshape((1, T)))
	BIC[k_cur - 1] = (2*L - 3*k_cur)*np.log(np.linalg.norm(Y - Y1)**2) + (5*k_cur + 1)*np.log(L)
	
	while k_cur < k_max:
		k_cur += 1
		Yt = np.copy(Y)
		for i in range(k_cur - 1):
			Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Yt.copy(), theta_hat[k_cur-1])
		dif = 1
		iter_num = 1
		num = 1
		while dif > epsilon:
			Yt = Y.copy()
			for i in range(k_cur):
				if i != num - 1:
					Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
# 			theta_hat[num-1], S_hat[num-1, :] = func_fft_max(Yt.copy(), fft_num, L)
			theta_hat[num-1], S_hat[num-1, :] = func_newton_search(Yt.copy(), theta_hat[num-1])
			if num == k_cur:
				if iter_num == max_iter: break
				iter_num += 1
				likelihood_v = np.linalg.norm(Yt - f_steer_vec(theta_hat[num-1], L).dot(S_hat[num-1, :].reshape((1, T))))**2
				dif = np.abs(likelihood_v - likelihood_pre)/ np.abs(likelihood_pre)
				likelihood_pre = likelihood_v.copy()
			num = np.mod(num, k_cur) + 1
		
		theta_hat_temp[:k_cur, k_cur - 1] = theta_hat[:k_cur].reshape((k_cur,))
		S_hat_temp[:k_cur, :, k_cur - 1] = S_hat[:k_cur, :]
		
		Y1 = f_steer_vec(theta_hat[:k_cur], L).dot(S_hat[:k_cur, :].reshape((k_cur, T)))
		BIC[k_cur - 1] = (2*L - 3*k_cur)*np.log(np.linalg.norm(Y - Y1)**2) + (5*k_cur + 1)*np.log(L)
	
	# BIC_min_v = np.min(BIC)
	BIC_index = np.argmin(BIC)
	
	if source_num_max == 0:
		theta_es = theta_hat_temp[:BIC_index + 1, BIC_index]
		S_es = S_hat_temp[:BIC_index + 1, :, BIC_index]
		source_num_es = BIC_index + 1
	else:
		theta_es = theta_hat[:k_cur]
		S_es = S_hat[:k_cur, :]
		source_num_es = k_cur
	
	angle = theta_es/np.pi*180
	power = np.mean(np.abs(S_es)**2, 1)
	
	return angle, power, source_num_es, S_es, BIC


# RELAX1(stop on angle differences)
def func_RELAX1(Y, fft_num = 1024, k_max = 5, epsilon_thre=1e-3):
	L, T = np.shape(Y)
	
	theta_hat = np.zeros((k_max, 1))
	BIC_v = np.zeros((k_max+1, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))
	
	k_cur = 1
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Y.copy(), fft_num, L)
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Y.copy(), theta_hat[k_cur-1])
	Y1 = f_steer_vec(theta_hat[k_cur - 1], L).dot(S_hat[k_cur - 1, :].reshape((1, T)))
	BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur-1].copy(), S_hat[0:k_cur-1, :].copy(), k_cur)
	while k_cur < k_max:
		k_cur += 1
		Yt = np.copy(Y)
		for i in range(k_cur - 1):
			Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Yt.copy(), theta_hat[k_cur-1])
		theta_hat_pre = theta_hat.copy()
		epsilon = 1
		num = 1
		while epsilon > epsilon_thre:
			Yt = Y.copy()
			for i in range(k_cur):
				if i != num - 1:
					Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
			theta_hat[num-1], S_hat[num-1, :] = func_fft_max(Yt.copy(), fft_num, L)
			theta_hat[num-1], S_hat[num-1, :] = func_newton_search(Yt.copy(), theta_hat[num-1])
			if num == k_cur:
# 				epsilon = np.abs(loglike_v - loglike_pre)/ np.abs(loglike_v)
				epsilon = np.linalg.norm(theta_hat - theta_hat_pre)/np.linalg.norm(theta_hat_pre)
				theta_hat_pre = theta_hat.copy()
			num = np.mod(num, k_cur) + 1
		BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur].copy(), S_hat[0:k_cur, :].copy(), k_cur)
	theta_es = theta_hat[:k_cur]
	S_es = S_hat[:k_cur, :]
	source_num_es = k_cur
	
	angle = theta_es/np.pi*180
	power = np.mean(np.abs(S_es)**2, 1)
	
	return angle, power, source_num_es, S_es, BIC_v.reshape((k_max+1,))

# RELAX_pro
def func_RELAX_pro(Y, fft_num = 1024, k_max = 5, max_iter = 30):
	L, T = np.shape(Y)
	
	theta_hat = np.zeros((k_max, 1))
	BIC_v = np.zeros((k_max+1, 1))
	S_hat = np.zeros((k_max, T)) + 1j*np.zeros((k_max, T))
	
	k_cur = 1
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Y.copy(), fft_num, L)
	theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Y.copy(), theta_hat[k_cur-1])
	Y1 = f_steer_vec(theta_hat[k_cur - 1], L).dot(S_hat[k_cur - 1, :].reshape((1, T)))
	loglike_pre = np.log(np.linalg.norm(Y - Y1)**2)
	BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur-1].copy(), S_hat[0:k_cur-1, :].copy(), k_cur)
	while k_cur < k_max:
		k_cur += 1
		Yt = np.copy(Y)
		for i in range(k_cur - 1):
			Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_fft_max(Yt.copy(), fft_num, L)
		theta_hat[k_cur-1], S_hat[k_cur-1, :] = func_newton_search(Yt.copy(), theta_hat[k_cur-1])
		epsilon = 1
		num = 1
		iter_num = 1
		while epsilon > 1e-5:
			Yt = Y.copy()
			for i in range(k_cur):
				if i != num - 1:
					Yt -= f_steer_vec(theta_hat[i], L).dot(S_hat[i, :].reshape((1, T)))
			theta_hat[num-1], S_hat[num-1, :] = func_fft_max(Yt.copy(), fft_num, L)
			theta_hat[num-1], S_hat[num-1, :] = func_newton_search(Yt.copy(), theta_hat[num-1])
			if num == k_cur:
				if iter_num == max_iter:
					break
				iter_num += 1
				loglike_v = np.log(np.linalg.norm(Yt - f_steer_vec(theta_hat[num-1], L).dot(S_hat[i, :].reshape((1, T))))**2)
				epsilon = np.abs(loglike_v - loglike_pre)/ np.abs(loglike_v)
				loglike_pre = loglike_v.copy()
			num = np.mod(num, k_cur) + 1
		BIC_v[k_cur-1] = func_BIC(Y.copy(), theta_hat[0:k_cur].copy(), S_hat[0:k_cur, :].copy(), k_cur)
	theta_es = theta_hat[:k_cur]
	S_es = S_hat[:k_cur, :]
	source_num_es = k_cur
	
	angle = theta_es/np.pi*180
	power = np.mean(np.abs(S_es)**2, 1)
	
	return angle, power, source_num_es, S_es, BIC_v.reshape((k_max+1,))
