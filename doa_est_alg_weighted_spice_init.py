import numpy as np
import numpy.matlib

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

def func_lmmse(Y, A, p):
	N, N_tot = np.shape(A)
	M = N_tot - N
	I_N = np.eye(N)
	
	R_inv = np.linalg.inv((A*p.T).dot(A.conj().T))
	Y_tilde = R_inv.dot(Y)
	S_hat = p.dot(A[:,:M].conj().T).dot(Y_tilde)
	return S_hat
	
def func_lmvue(Y, A, p):
	N, N_tot = np.shape(A)
	M = N_tot - N
	I_N = np.eye(N)
	
	R_inv = np.linalg.inv((A*p.T).dot(A.conj().T))
	Y_tilde = R_inv.dot(Y)
	Temp = R_inv.dot(A)
	Temp = A.conj()*Temp
	temp = np.sum(Temp, 0).reshape(N_tot, 1)
	Deno = A.conj().T.dot(Y_tilde)
	S_hat = Deno/temp
	return S_hat

def iaa_init(X, A_bar, max_iter, epsilon = 1e-3, p_hat_init = 0):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	
	s_hat = (A_bar.conj().T).dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, ))
	p_hat = np.sum(s_hat*s_hat.conj(), 1).reshape((G_tot, 1))/T

	if not isinstance(p_hat_init, int):
		p_hat = p_hat_init.copy().reshape((G_tot, 1))
		
	p_hat_pre = p_hat.copy()
	iter_num = 1
	dif = 1
	while dif > epsilon:
		R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T) 
		R_invA = np.linalg.inv(R_hat).dot(A_bar)
		part1 = R_invA.conj().T.dot(X)
		part2 = np.sum(R_invA.conj()*A_bar, 0).reshape((G_tot, 1))
		s_hat = part1/part2
		p_hat = np.sum(s_hat*s_hat.conj(), 1)/T
	
		if iter_num == max_iter: break
		dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
		p_hat_pre = p_hat.copy()
		iter_num += 1
		
	return abs(p_hat)
	
def slim_init(y_noisy, A, Iter_no, epsilon = 1e-3, p_hat_init = 0):
	L, G = np.shape(A)
	
	s_hat = A.conj().T.dot(y_noisy)/np.sum(A.conj()*A, 0).reshape(G, 1)
	p = np.abs(s_hat)**2
	
	if not isinstance(p_hat_init, int):
		p = p_hat_init.copy().reshape((G, 1))
		
	theta = np.max(np.abs(y_noisy))*1e-6
	eta = np.max(np.abs(p))*1e-6
	
	p_pre = p.copy()
	iter_num = 1
	dif = 1
	while dif > epsilon:
		tmp = numpy.matlib.repmat(p.T, L, 1)
		Ap = A*tmp
		R = Ap.dot(A.conj().T) + eta*np.eye(L)
		Ri = np.linalg.inv(R)
		
		s_hat = p*(A.conj().T.dot(Ri.dot(y_noisy)))
		p = np.abs(s_hat)**2 + theta
		eta = np.linalg.norm(y_noisy - A.dot(s_hat))**2/L
		
		if iter_num == Iter_no: break
		dif = np.linalg.norm(p - p_pre)/np.linalg.norm(p_pre)
		p_pre = p.copy()
		iter_num += 1
	p -= theta
	return p
	
def likes_init(y_noisy, A, Iter_no, epsilon = 1e-3, p_hat_init = 0):
	L, T = np.shape(y_noisy)
	L, G = np.shape(A)
	
	s_hat = A.conj().T.dot(y_noisy)/np.sum(A.conj()*A, 0).reshape(G, 1)
	p = np.abs(s_hat)**2
	
	if not isinstance(p_hat_init, int):
		p = p_hat_init.copy().reshape((G, 1))
		s_hat = func_lmvue(y_noisy, A, p)

	p_pre = p.copy()
	iter_num = 1
	dif = 1
	while dif > epsilon:
		eta = np.linalg.norm(y_noisy - A.dot(s_hat))**2/L
		tmp = numpy.matlib.repmat(p.T, L, 1)
		Ap = A*tmp
		R = Ap.dot(A.conj().T) + eta*np.eye(L)
		Ri = np.linalg.inv(R)
		
		s_hat = p*(A.conj().T.dot(Ri.dot(y_noisy)))
		R_invA = Ri.dot(A)
		w = np.sum(R_invA.conj()*A, 0).reshape((G, 1))
		p = np.abs(s_hat)/np.real(np.sqrt(w))
		
		if iter_num == Iter_no: break
		dif = np.linalg.norm(p - p_pre)/np.linalg.norm(p_pre)
		p_pre = p.copy()
		iter_num += 1
	return p 

def spice_init(X, A_bar, max_iter, epsilon = 1e-3, p_hat_init = 0):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = A_bar.conj().T.dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = (np.sum(s_hat*s_hat.conj(), 1)/T).reshape((G_tot, 1))
	
	if not isinstance(p_hat_init, int):
		p_hat = p_hat_init.copy().reshape((G_tot, 1))
		
	p_hat_pre = p_hat.copy()
	
	iter_num = 1

	if T == 1:
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			z = np.linalg.inv(R_hat).dot(X)
			temp_v = A_bar.conj().T.dot(z)
			p_hat = p_hat*abs(temp_v)
			if iter_num == max_iter: break
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
			ind += 1
			p_hat_pre = p_hat.copy()
			iter_num += 1
		s_hat = func_lmvue(X, A_bar, p_hat)
		power = abs(s_hat)**2
	elif 1 < T < L:
		R_hat_samp = X.dot(X.conj().T)/T
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			Z = np.linalg.inv(R_hat).dot(R_hat_samp)
			temp_M = (A_bar.conj().T).dot(Z)
			temp_v = np.sqrt(np.sum(temp_M*(temp_M.conj()), 1)).reshape((G_tot, 1))
			p_hat = p_hat*abs(temp_v)
			
			if iter_num == max_iter: break
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
			ind += 1
			p_hat_pre = p_hat.copy()
			iter_num += 1
		S_hat = func_lmvue(X, A_bar, p_hat)
		power = np.mean(abs(S_hat)**2, 1)
	else: # T >= L
		R_hat_samp = X.dot(X.conj().T)/T
		V, U = np.linalg.eig(R_hat_samp)
		# sorted_indices = np.argsort(V)
		R_hat_square = U.dot(np.diag(np.sqrt(V))).dot(U.conj().T)
		R_hat_inv = np.linalg.inv(R_hat_samp)
		w_k = np.real(sum(A_bar.conj()*(R_hat_inv.dot(A_bar)))).reshape((G_tot, 1))
		w_k_sqrt = np.sqrt(w_k)
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			Z = np.linalg.inv(R_hat).dot(R_hat_square)
			temp_M = (A_bar.conj().T).dot(Z)
			temp_v = np.sqrt(np.sum(temp_M*(temp_M.conj()), 1)).reshape((G_tot, 1))
			p_hat = p_hat*abs(temp_v)/w_k
			if iter_num == max_iter: break
			ind += 1
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
			p_hat_pre = p_hat.copy()
			iter_num += 1
		S_hat = func_lmvue(X, A_bar, p_hat)
		power = np.mean(abs(S_hat)**2, 1)
	return power
			

def likes_init_s_hat(y_noisy, A, Iter_no, epsilon = 1e-3, s_hat_init = 0):
	L, T = np.shape(y_noisy)
	L, G = np.shape(A)
	
	s_hat = A.conj().T.dot(y_noisy)/np.sum(A.conj()*A, 0).reshape(G, 1)

	if not isinstance(s_hat_init, int):
		s_hat = s_hat_init.copy().reshape((G, 1))

	p = np.abs(s_hat)**2
	p_pre = p.copy()
	iter_num = 1
	dif = 1
	while dif > epsilon:
		eta = np.linalg.norm(y_noisy - A.dot(s_hat))**2/L
		tmp = numpy.matlib.repmat(p.T, L, 1)
		Ap = A*tmp
		R = Ap.dot(A.conj().T) + eta*np.eye(L)
		Ri = np.linalg.inv(R)
		
		s_hat = p*(A.conj().T.dot(Ri.dot(y_noisy)))
		R_invA = Ri.dot(A)
		w = np.sum(R_invA.conj()*A, 0).reshape((G, 1))
		p = np.abs(s_hat)/np.real(np.sqrt(w))
		
		if iter_num == Iter_no: break
		dif = np.linalg.norm(p - p_pre)/np.linalg.norm(p_pre)
		p_pre = p.copy()
		iter_num += 1
	return p, s_hat 


def iaa_relax_init(X, A_bar, iaa_max_iter, target_num, p_hat_init):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = (A_bar.conj().T).dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = np.sum(s_hat*s_hat.conj(), 1).reshape((G_tot, 1))/T
    
	if not isinstance(p_hat_init, int):
		p_hat = p_hat_init.copy().reshape((G_tot, 1))
        
	for i in range(iaa_max_iter):
		R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T) + 1e-8*np.eye(L)
		R_invA = np.linalg.inv(R_hat).dot(A_bar)
		part1 = R_invA.conj().T.dot(X)
		part2 = np.sum(R_invA.conj()*A_bar, 0).reshape((G_tot, 1))
		s_hat = part1/part2
		p_hat = np.sum(s_hat*s_hat.conj(), 1)/T
	
	power = p_hat[:G]
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
	
	return angle, power, p_hat