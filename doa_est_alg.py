import numpy as np
import numpy.matlib

def dbf(X, grid_size):
	L, T = np.shape(X)
	return np.fft.fftshift(np.sum(np.abs(np.fft.fft(X, grid_size, 0)), 1)/(L*T))**2


def DAS(X, A_bar):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = A_bar.conj().T.dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = (np.sum(s_hat*s_hat.conj(), 1)/T).reshape((G_tot, 1))
	return abs(p_hat[:G])
    

def capon(X, grid_size):
	L, T = np.shape(X)
	R = X.dot(X.conj().T)/T
	R_inv = np.linalg.inv(R)
	theta_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint = False)
	A = np.exp(1j*np.array([range(L)]).T.dot(theta_grid.reshape((1, grid_size))))
	p_hat = 1/np.abs(np.diag(A.conj().T.dot(R_inv).dot(A)))
	return p_hat

def iaa(X, grid_size, max_iter):
	L, T = np.shape(X)
	theta_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint = False)
	A = np.exp(-1j*np.array([range(L)]).T.dot(theta_grid.reshape((1, grid_size))))

	p_hat_old = np.fft.fftshift(np.sum(np.abs(np.fft.fft(X, grid_size, 0)), 1)/(L*T))**2
	n_iter = 1
	while n_iter <= max_iter:
		R = A.dot(np.diag(p_hat_old)).dot(A.conj().T)
		R_inv = np.linalg.inv(R)
		p_hat = (np.mean(np.abs(A.conj().T.dot(R_inv).dot(X)), 1)/np.abs(np.diag(A.conj().T.dot(R_inv).dot(A))))**2
		if np.linalg.norm(p_hat - p_hat_old)/np.linalg.norm(p_hat_old) < 0.001: break
		p_hat_old = p_hat
		n_iter += 1
	return p_hat

def iaa1(X, A_bar, max_iter):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = (A_bar.conj().T).dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = np.sum(s_hat*s_hat.conj(), 1).reshape((G_tot, 1))/T
	for i in range(max_iter):
		R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T) 
		R_invA = np.linalg.inv(R_hat).dot(A_bar)
		part1 = R_invA.conj().T.dot(X)
		part2 = np.sum(R_invA.conj()*A_bar, 0).reshape((G_tot, 1))
		s_hat = part1/part2
		p_hat = np.sum(s_hat*s_hat.conj(), 1)/T
	return abs(p_hat[:G])
	
def slim(X, A_bar, max_iter):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = A_bar.conj().T.dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = (np.sum(s_hat*s_hat.conj(), 1)/T).reshape((G_tot, 1))
	p_hat_pre = p_hat.copy()
	diff, idx, epsilon = 1, 0, 1e-3
	
	while idx < max_iter and diff > epsilon:
		w = 1/p_hat
		R_hat = (A_bar*(p_hat.T)).dot(A_bar.conj().T)
		z = (A_bar.conj().T.dot(np.linalg.inv(R_hat).dot(X)))
		s_hat = p_hat*z
		q = np.mean(abs(s_hat)**2, 1)
		max_X = max(abs(X))
		p_hat = np.mean(abs(s_hat)**2, 1).reshape((G_tot, 1))/w**0.5 + 1e-4*max_X
		diff = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
		idx += 1
		p_hat_pre = p_hat.copy()
	return abs(p_hat[:G])

def slim_q(X, A, q):
	L, T = np.shape(X)
	L, G = np.shape(A)
    
	x0 = A.conj().T.dot(X)/np.sum(A.conj()*A, 0).reshape((G, 1))
	p_hat = abs(x0)**(2 - q)
	iter_max =50
	eta = 1/L*np.linalg.norm(X-A.dot(x0))**2
	eta=0.001*max(abs(p_hat))
	s = np.zeros((G, 1))
	epllison=0.0001*max(abs(X))    
	for i in range(iter_max):
		R_hat = (A*p_hat.T).dot(A.conj().T) + eta*np.eye(L)
		z = s[:]
		s = p_hat*A.conj().T.dot(np.linalg.inv(R_hat).dot(X))
		eta = 1/L*np.linalg.norm(X-A.dot(s))**2
		p_hat = abs(s)**(2-q)+epllison
		dif = abs(np.linalg.norm(s) - np.linalg.norm(z))/np.linalg.norm(s)
		if dif < 0.001: break
	return p_hat
	
def likes(X, A_bar):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = A_bar.conj().T.dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = (np.sum(s_hat*s_hat.conj(), 1)/T).reshape((G_tot, 1))
	p_hat_pre = p_hat.copy()
	epsilon = 1e-3
	if T == 1:
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			R_hat_inv = np.linalg.inv(R_hat)
			w_k = np.real(sum(A_bar.conj()*(R_hat_inv.dot(A_bar)))).reshape((G_tot, 1))
			w_k_sqrt = np.sqrt(w_k)
			while ind < 50:
				R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
				z = np.linalg.inv(R_hat).dot(X)
				temp_v = A_bar.conj().T.dot(z)
				s_hat = p_hat*temp_v
				p_hat = abs(s_hat)/w_k_sqrt
				ind += 1
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
			p_hat_pre = p_hat.copy()
		s_hat = func_lmvue(X, A_bar, p_hat)
		power = abs(s_hat[:G])**2
	else: # T > 1
		R_hat_samp = X.dot(X.conj().T)/T
		V, U = np.linalg.eig(R_hat_samp)
		# sorted_indices = np.argsort(V)
		R_hat_square = U.dot(np.diag(np.sqrt(V))).dot(U.conj().T)
		dif = 1
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			R_hat_inv = np.linalg.inv(R_hat)
			w_k = np.real(sum(A_bar.conj()*(R_hat_inv.dot(A_bar)))).reshape((G_tot, 1))
			w_k_sqrt = np.sqrt(w_k)
			ind = 0
			while ind < 30:
				R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
				Z = np.linalg.inv(R_hat).dot(R_hat_square)
				temp_M = (A_bar.conj().T).dot(Z)
				temp_v = np.sqrt(np.sum(temp_M*(temp_M.conj()), 1)).reshape((G_tot, 1))
				p_hat = p_hat*abs(temp_v)/w_k_sqrt
				ind += 1
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
			p_hat_pre = p_hat.copy()
		power = abs(p_hat[:G])
	return power
	
def func_slim_power(y_noisy, A, Iter_no):
	L, G = np.shape(A)
	s_hat = A.conj().T.dot(y_noisy)/np.sum(A.conj()*A, 0).reshape(G, 1)
	p = np.abs(s_hat)**2
	theta = np.max(np.abs(y_noisy))*1e-6
	eta = np.max(np.abs(p))*1e-6
	for i in range(Iter_no):
		tmp = numpy.matlib.repmat(p.T, L, 1)
		Ap = A*tmp
		R = Ap.dot(A.conj().T) + eta*np.eye(L)
		Ri = np.linalg.inv(R)
		
		s_hat = p*(A.conj().T.dot(Ri.dot(y_noisy)))
		p = np.abs(s_hat)**2 + theta
		eta = np.linalg.norm(y_noisy - A.dot(s_hat))**2/L
	p -= theta
	return p[:G-L]
	
def func_likes_power(y_noisy, A, Iter_no):
	L, T = np.shape(y_noisy)
	L, G = np.shape(A)
	s_hat = A.conj().T.dot(y_noisy)/np.sum(A.conj()*A, 0).reshape(G, 1)
	p = np.abs(s_hat)**2
	w = 1/np.sum(s_hat*s_hat.conj(), 1)/T

	for i in range(Iter_no):
		eta = np.linalg.norm(y_noisy - A.dot(s_hat))**2/L
		tmp = numpy.matlib.repmat(p.T, L, 1)
		Ap = A*tmp
		R = Ap.dot(A.conj().T) + eta*np.eye(L)
		Ri = np.linalg.inv(R)
		
		s_hat = p*(A.conj().T.dot(Ri.dot(y_noisy)))
		R_invA = Ri.dot(A)
		w = np.sum(R_invA.conj()*A, 0).reshape((G, 1))
		p_pre = p.copy()
		p = np.abs(s_hat)/np.real(np.sqrt(w))
	return p[:G-L] 
    
def spice(X, A_bar):
	L, T = np.shape(X)
	G_tot = len(A_bar[0])
	G = G_tot - L
	s_hat = A_bar.conj().T.dot(X)/np.sum(A_bar.conj()*A_bar, 0).reshape((G_tot, 1))
	p_hat = (np.sum(s_hat*s_hat.conj(), 1)/T).reshape((G_tot, 1))
	p_hat_pre = p_hat.copy()
	epsilon = 1e-3

	if T == 1:
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			z = np.linalg.inv(R_hat).dot(X)
			temp_v = A_bar.conj().T.dot(z)
			p_hat = p_hat*abs(temp_v)
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
			ind += 1
			p_hat_pre = p_hat.copy()
		s_hat = func_lmvue(X, A_bar, p_hat)
		power = abs(s_hat[:G])**2
	elif 1 < T < L:
		R_hat_samp = X.dot(X.conj().T)/T
		dif, ind = 1, 0
		while dif > epsilon:
			R_hat = (A_bar*p_hat.T).dot(A_bar.conj().T)
			Z = np.linalg.inv(R_hat).dot(R_hat_samp)
			temp_M = (A_bar.conj().T).dot(Z)
			temp_v = np.sqrt(np.sum(temp_M*(temp_M.conj()), 1)).reshape((G_tot, 1))
			p_hat = p_hat*abs(temp_v)
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
			ind += 1
			p_hat_pre = p_hat.copy()
		S_hat = func_lmvue(X, A_bar, p_hat)
		power = np.mean(abs(S_hat[:G, :])**2, 1)
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
			ind += 1
			dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat)
			p_hat_pre = p_hat.copy()
		S_hat = func_lmvue(X, A_bar, p_hat)
		power = np.mean(abs(S_hat[:G, :])**2, 1)
	return power
			
		
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

def peak_selector(p_hat, K_max):
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
    return [idx_hat, [abs(p_hat[i]) for i in idx_hat]] 
