import numpy as np
import numpy.matlib

def space_smooth(Y, L_sub = 0, flag = 1):
	# sample array
	# L_sub: subarray length
	# flag: 0-front smoothing; 1-front and back smoothing
	
	L, T = np.shape(Y)
	if L_sub == 0: L_sub = L//2
	N_sub = L - L_sub + 1
	
	R_sub = np.zeros((L_sub, L_sub, N_sub)) + 1j*np.zeros((L_sub, L_sub, N_sub))
	
	for i in range(N_sub):
		R_sub[:, :, i] = Y[i:i + L_sub, :].dot(Y[i:i + L_sub, :].conj().T)/T
	
	if flag == 0:
		R = np.mean(R_sub, 2)
	else:
		J = np.fliplr(np.eye(L_sub))
		Rf = np.mean(R_sub, 2)
		R = 0.5*(Rf + J.dot(Rf.T).dot(J))
	return R
	
def steer_vecs(angles, L):
	l = np.array([range(L)]).T
	k = np.size(angles)
	return np.exp(-1j*np.pi*l.dot(np.sin(angles.reshape((1, k)))))
	
	
def peak_selector(p_hat, K_max):
    G = len(p_hat)
    p_peak = np.array([0]*G)
    
    for k in range(1, G - 1):
        if p_hat[k] > p_hat[k + 1] and p_hat[k] > p_hat[k - 1]:
            p_peak[k] = 1
            
    p_idx_hat = (p_hat.reshape(G, 1)*p_peak.reshape(G, 1)).reshape(G, )
    idx_sort = sorted(range(len(p_hat)), key=lambda k: p_idx_hat[k], reverse=True)
    if len(idx_sort) <= K_max: return sorted(idx_sort)
    idx_hat = sorted(idx_sort[:K_max])
    return idx_hat, p_hat[idx_hat]
	
	
def music(X, G, K, L_sub = 0):
	L, T = np.shape(X)
	
	if L_sub == 0: L_sub = int(L//2)
	R = space_smooth(X, L_sub, 1)
	
	f_grid = np.linspace(-1, 1, G, endpoint = False)
	A = np.exp(-1j*np.pi*np.array([range(L_sub)]).T.dot(f_grid.reshape((1, G))))
	
	# eigenvalue decomposition
	vals, vecs = np.linalg.eig(R)
	idx = np.argsort(vals)
	Un = vecs[:, idx[:L_sub - K]]
	
	# p_hat = 1/np.abs(np.diag(A.conj().T.dot(Un.dot(Un.conj().T)).dot(A)))
	p_hat = 1/np.sum(np.abs((A.conj().T.dot(Un)))**2, 1)
	peak_idx = peak_selector(p_hat, K)[0]
	
	angle_est = np.arcsin(f_grid[peak_idx])
	
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est, p_hat
	

def root_music(X, K, L_sub = 0):
	L, T = np.shape(X)
	
	if L_sub == 0: L_sub = int(L//2)
	R = space_smooth(X, L_sub, 1)
	
	# eigenvalue decomposition
	vals, vecs = np.linalg.eig(R)
	idx = np.argsort(vals)
	Un = vecs[:, idx[:L_sub - K]].copy()
	
	# multinomial coefficients
	C = Un.dot(Un.conj().T)
	coeff = np.zeros(L_sub - 1,) + 1j*np.zeros(L_sub - 1,)
	for ii in range(L_sub - 1):
		coeff[ii] = np.sum(np.diag(C, ii + 1))
		
	coeff = np.append(np.append(coeff[::-1].copy(), np.sum(np.diag(C))), coeff.conj().copy())
	
	# solve equation to get roots
	z = np.roots(coeff)
	
	# find the K points inside and closest to the unit circle
	len_z = np.size(z)
	mask = np.zeros([len_z,])
	for ii in range(len_z):
		absz = np.abs(z[ii])
		if absz > 1:
			mask[ii] = float('inf')
		else:
			mask[ii] = 1 - absz		
	idx = np.argsort(mask)[:K]
	z = z[idx]
	
	angle_est = np.sort(np.arcsin(-np.angle(z)/np.pi))
	
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est
				

def esprit(X, K, use_tls = 1, L_sub = 0):
	# use_tls: 0-local LS; 1-total LS
	L, T = np.shape(X)
	
	if L_sub == 0: L_sub = int(L//2)
	R = space_smooth(X, L_sub, 1)
	
	ds = 1 # The two sub-arrays are separated by 1 antenna
	
	# eigenvalue decomposition
	vals, vecs = np.linalg.eig(R)
	idx = np.argsort(vals)
	Us = vecs[:, idx[- K:]].copy()
	Us1, Us2 = Us[:-ds, :].copy(), Us[ds:, :].copy()
	
	if use_tls == 0:
		Phi = np.linalg.inv(Us1.conj().T.dot(Us1)).dot(Us1.conj().T.dot(Us2))
		
	else:
		C = np.append(Us1, Us2, axis = 1)
		C = C.conj().T.dot(C)
		C = 0.5*(C + C.conj().T)
		
		vals, vecs = np.linalg.eig(C)
		idx = np.argsort(-np.real(vals))
		V = vecs[:, idx].copy()
		
		V12 = V[:K, K:]
		V22 = V[K:, K:]
		Phi = -V12.dot(np.linalg.inv(V22))
	
	vals = np.linalg.eig(Phi)[0]
	angle_est = np.sort(np.arcsin(-np.angle(vals)/(ds*np.pi)))
	
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est


def compute_G_inv_S(G, S, L, K):
	# function used in mode
	Z = S.copy()
	for j in range(L - K):
		Z[j, :] = Z[j, :]/G[j, j]
		for i in range(j + 1, min(j + K + 1, L - K)):
			Z[i, :] = Z[i, :] - G[i, j]*Z[j, :]
	return Z
		
	
def compute_H(G, Us, L, K):
	# function used in mode
	S = np.zeros((L - K, K + 1, K)) + 1j*np.zeros((L - K, K + 1, K))
	H = np.zeros((K*(L - K), K + 1)) + 1j*np.zeros((K*(L - K), K + 1))
	
	for ii in range(K):
		for jj in range(L - K):
			S[jj, :, ii] = Us[jj:(jj + K + 1), ii].T.copy()
		H[ii:(ii + L - K), :] = compute_G_inv_S(G, np.flip(S[:, :, ii], 1), L, K)
	return H
	
def compute_C(K, L, b):
	# function used in mode
	rho = np.zeros(K + 1,) + 1j*np.zeros(K + 1,)
	
	for k in range(K + 1):
		rho[k] = 0
		for l in range(K - k + 1):
			rho[k] = rho[k] + b[l].conj()*b[l+k]
			
	C = np.zeros((L - K, L - K)) + 1j*np.zeros((L - K, L - K))
	for i in range(L - K):
		for j in range(L - K):
			if abs(i - j) <= K and i >= j:
				C[i, j] = rho[i-j]
			elif abs(i - j) <= K and i < j:
				C[i, j] = (rho[j - i]).conj()
	return C
	
def compute_G(K, L, C):
	# function used in mode
	G = C.copy()
	for j in range(L - K):
		for l in range(max(0, j - K), j):
			G[j, j] -= np.abs(G[j, l])**2
		G[j, j] = np.sqrt(G[j, j])
		for i in range(j+1, min(j+K+1, L-K)):
			for l in range(max(0, j - K), j):
				G[i, j] = G[i, j] - G[i, l]*G[j, l].conj()
			G[i, j] = G[i, j]/G[j, j]
	return G
	
def compute_W(K):
	# function used in mode
	W1 = np.array([[1, 1j], [1, -1j]])
	W2 = np.array([[1, 1j, 0], [0, 0, 1], [1, -1j, 0]])
	if K%2 == 0:
		W = W2.copy()
		w = int(K / 2)
	else:
		W = W1.copy()
		w = int((K - 1) / 2)
	
	if K > 2:
		for i in range(w):
			w1, w2 = np.shape(W)
			W_tmp = np.zeros((w1 + 2, w2 + 2)) + 1j*np.zeros((w1 + 2, w2 + 2))
			W_tmp[1:-1, 2:] = W.copy()
			W_tmp[0, :2] = np.array([1, 1j])
			W_tmp[-1, :2] = np.array([1, -1j])
			W = W_tmp.copy()
	return W
	
def compute_Y(K, L, T, X):
	# function used in iqml
	Y = np.zeros((L - K, K + 1, T)) + 1j*np.zeros((L - K, K + 1, T)) 
	for t in range(T):
		for i in range(K + 1):
			Y[:, i, t] = X[K-i:L-i, t].copy()
	return Y

def compute_Psi(K, L, T, Y, G):
	# function used in mode
	Z = Y.copy()
	Psi = np.zeros((K + 1, K + 1, T)) + 1j*np.zeros((K + 1, K + 1, T)) 
	for t in range(T):
		for j in range(L - K):
			Z[j, :, t] = Z[j, :, t]/G[j, j]
			for i in range(j+1, min(j + K + 1, L - K)):
				Z[i, :, t] = Z[i, :, t] - G[i, j]*Z[j, :, t]
		Psi[:, :, t] = Z[:, :, t].conj().T.dot(Z[:, :, t])
	return Psi
				
	
			
	
	
	
def mode(X, K, L_sub = 0, iter_limit = 5):
	
	L, T = np.shape(X)
	
	if L_sub == 0: L_sub = int(L//2)
	R = space_smooth(X, L_sub, 1)
	
	W = compute_W(K)
	
	# eigenvalue decomposition
	vals, vecs = np.linalg.eigh(R)
	idx = np.argsort(-np.abs(vals))
	Us = vecs[:, idx[:K]].copy()
	Es = vals[idx[:K]]
	
	sigma2 = np.sum(vals[idx[K:]])/(L_sub - K)
	
	S = Us.dot((np.diag((Es - sigma2*np.ones(K))**2/Es))**0.5)

	C = np.eye(L_sub - K)

	b_tmp = np.ones(K + 1)
	
	for i in range(iter_limit):
		G = compute_G(K, L_sub, C)
		H = compute_H(G, S, L_sub, K)
		Ohm = np.vstack((np.real(H.dot(W)), np.imag(H.dot(W))))
		Q, R0 = np.linalg.qr(Ohm[:, 1:], 'complete')
		
		beta = np.vstack((np.array([[1]]), -np.linalg.inv(R0[:K, :]).dot((Q[:, :K].conj().T).dot(Ohm[:, :1]))))

		b = W.dot(beta)
		b = b.reshape((np.size(b),))

		if np.linalg.norm(b-b_tmp)/np.linalg.norm(b_tmp) < 1e-3:
			break
		b_tmp = b.copy()
		
		C = compute_C(K, L_sub, b)
		
	z = np.roots(b)
	angle_est = np.sort(-np.arcsin(np.angle(z)/np.pi))
	
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est


def iqml(X, K, iter_limit = 10):
	
	L, T = np.shape(X)
	R = X.dot(X.conj().T)/T
	Y = compute_Y(K, L, T, X)
	W = compute_W(K)
	
	C = np.eye(L - K)
	iter_limit = 5
	b_tmp = np.ones(K + 1)
	for i in range(iter_limit):
		G = compute_G(K, L, C)
		Psi = compute_Psi(K, L, T, Y, G)
		Ohm = np.real(W.conj().T.dot(np.sum(Psi, 2)).dot(W))
		beta = np.vstack((np.array([[1]]), -np.linalg.inv(Ohm[1:, 1:]).dot(Ohm[1:, :1])))
		
		b = W.dot(beta)
		b = b.reshape((np.size(b),))
		if np.linalg.norm(b-b_tmp)/np.linalg.norm(b_tmp) < 1e-3:
			break
		b_tmp = b.copy()
		
		C = compute_C(K, L, b)
	
	z = np.roots(b)
	angle_est = np.sort(-np.arcsin(np.angle(z)/np.pi))
	
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est
	
	
	
		
		
def em(X, K, G, iter_max, angles_pre = 0):
	
	L, T = np.shape(X)
	
	angle_grid = np.arcsin(np.linspace(-1, 1, G, endpoint = False))
	A0 = steer_vecs(angle_grid, L)
	
	# estimates initialization
	if angles_pre == 0:
		# dbf
		X_fft = np.fft.fftshift(np.fft.fft(X.conj(), G, 0)/L, 0)
		p_hat = np.mean(np.abs(X_fft)**2, 1)
		peak_idx = peak_selector(p_hat, K)[0]
		angles_pre = angle_grid[peak_idx]
	elif angles_pre == 1:
		# capon
		R_smooth = space_smooth(X, int(L//2), 1)
		A_smooth = steer_vecs(angle_grid, int(L//2))
		p_hat = 1/np.abs(np.diag(A_smooth.conj().T.dot(np.linalg.inv(R_smooth)).dot(A_smooth)))
		peak_idx = peak_selector(p_hat, K)[0]
		angles_pre = angle_grid[peak_idx]
	
	A_est = steer_vecs(angles_pre, L)
	S_pre = A_est.conj().T.dot(X)/L
	sigma2_pre = np.linalg.norm(X - A_est.dot(S_pre))**2/(L*T)
	sigma2_vec_pre = np.ones(L)*sigma2_pre/L
	loglikeli_tmp = L*np.log(sigma2_pre) + np.linalg.norm(X - A_est.dot(S_pre))**2/(T*sigma2_pre)
	
	angles_est = np.zeros(K)
	S_hat = np.zeros((K,T)) + 1j*np.zeros((K,T))
	sigma2_vec_est = np.zeros(K)
	iter_num = 1
	
	while iter_num < iter_max:
		for i in range(K):
			a_pre = steer_vecs(angles_pre[i], L)
			
			Y = a_pre*S_pre[i, :] + sigma2_vec_pre[i]/sigma2_pre*(X - A_est.dot(S_pre))
			R = Y.dot(Y.conj().T)/T + sigma2_vec_pre[i]**2/sigma2_pre*np.eye(L)
			
			idx = np.argmax(np.abs(np.diag(A0.conj().T.dot(R).dot(A0))))
			angles_est[i] = angle_grid[idx]
			
			a = steer_vecs(angles_est[i], L)
			S_hat[i,:] = a.conj().T.dot(Y)/L
			
			sigma2_vec_est[i] = np.linalg.norm(Y - a.dot(S_hat[i:i+1,:]))**2/L
		
		sigma2_est = np.sum(sigma2_vec_est)
		A_est = steer_vecs(angles_est, L)
		loglikeli = L*np.log(sigma2_est) + np.linalg.norm(X - A_est.dot(S_hat))**2/(T*sigma2_est)
		
		if np.abs(loglikeli - loglikeli_tmp)/np.abs(loglikeli_tmp) < 1e-3:
			break
		else:
			angles_pre = angles_est.copy()
			S_pre = S_hat.copy()
			sigma2_vec_pre = sigma2_vec_est.copy()
			sigma2_pre = sigma2_est
			loglikeli_tmp = loglikeli
			iter_num += 1
			
	angles_est = np.sort(angles_est)
	power_est = np.mean(np.abs(S_hat)**2, 1)
	angle_est = angles_est/np.pi*180

	return angle_est, power_est
	
def sage(X, K, G, iter_max, angles_est = 0):
	
	L, T = np.shape(X)
	
	angle_grid = np.arcsin(np.linspace(-1, 1, G, endpoint = False))
	A0 = steer_vecs(angle_grid, L)
	
	# estimates initialization
	if angles_est == 0:
		# dbf
		X_fft = np.fft.fftshift(np.fft.fft(X.conj(), G, 0)/L, 0)
		p_hat = np.mean(np.abs(X_fft)**2, 1)
		peak_idx = peak_selector(p_hat, K)[0]
		angles_est = angle_grid[peak_idx]
	elif angles_est == 1:
		# capon
		R_smooth = space_smooth(X, int(L//2), 1)
		A_smooth = steer_vecs(angle_grid, int(L//2))
		p_hat = 1/np.abs(np.diag(A_smooth.conj().T.dot(np.linalg.inv(R_smooth)).dot(A_smooth)))
		peak_idx = peak_selector(p_hat, K)[0]
		angles_est = angle_grid[peak_idx]
	
	A_est = steer_vecs(angles_est, L)
	S_est = A_est.conj().T.dot(X)/L
	sigma2_est = np.linalg.norm(X - A_est.dot(S_est))**2/(L*T)
	loglikeli_tmp = L*np.log(sigma2_est) + np.linalg.norm(X - A_est.dot(S_est))**2/(T*sigma2_est)
	
	iter_num = 1
	while iter_num < iter_max:
		for i in range(K):
			
			Z = steer_vecs(angles_est[i], L)*S_est[i, :] + X - A_est.dot(S_est)
			C = Z.dot(Z.conj().T)/T + sigma2_est*np.eye(L)
			
			idx = np.argmax(np.abs(np.diag(A0.conj().T.dot(C).dot(A0))))
			angles_est[i] = angle_grid[idx]
			
			S_est[i,:] = steer_vecs(angles_est[i], L).conj().T.dot(Z)/L
			
			A_est = steer_vecs(angles_est, L)
			
			sigma2_est = np.linalg.norm(X - A_est.dot(S_est))**2/L
		
		loglikeli = L*np.log(sigma2_est) + np.linalg.norm(X - A_est.dot(S_est))**2/(T*sigma2_est)
		
		if np.abs(loglikeli - loglikeli_tmp)/np.abs(loglikeli_tmp) < 1e-3:
			break
		else:
			loglikeli_tmp = loglikeli
			iter_num += 1
			
	angles_est = np.sort(angles_est)
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angles_est/np.pi*180

	return angle_est, power_est
	
	
def update_proj_matrix(P, A):
	# function used in ap
	I = np.eye(np.shape(A)[0])
	Aa = (I - P).dot(A)
	b = Aa/np.sqrt(np.sum(np.abs(Aa)**2, 0))
	return b
	
	
def ap(X, K, G, iter_max=30, epsilon=1e-3):
	L, T = np.shape(X)
	
	theta_grid = np.linspace(-np.pi/2, np.pi/2, G, endpoint = False)
	angle_grid = np.arcsin(theta_grid/np.pi)
	
	A = steer_vecs(angle_grid, L)
	angle_0 = np.zeros(K)
	angle_idx = np.array([], dtype = int)
	for i in range(K):
		if i == 0:
			Proj = 0
		else:
			A_est = steer_vecs(angle_0[:i], L)
			Proj = A_est.dot(np.linalg.inv(A_est.conj().T.dot(A_est))).dot(A_est.conj().T)
		
		b = update_proj_matrix(Proj, A)
		cost_f = np.mean(np.abs(b.conj().T.dot(X)), 1)
		if i > 0:
			cost_f[angle_idx] = float('-inf')
		idx_max = np.argmax(cost_f)
		angle_idx = np.append(angle_idx, idx_max)
		angle_0[i] = angle_grid[idx_max]
	
	angle_pre = angle_0.copy()
	angle_est = angle_pre.copy()

	if K > 1:
		cost_pre = cost_f[idx_max]
		dif = 1
		iter_num = 1
		while dif > epsilon:
			for i in range(K):
				A_proj = steer_vecs(np.append(angle_est[:i], angle_est[i + 1:]), L)
				Proj = A_proj.dot(np.linalg.inv(A_proj.conj().T.dot(A_proj))).dot(A_proj.conj().T)
				b = update_proj_matrix(Proj, A)
				cost_f = np.mean(np.abs(b.conj().T.dot(X)), 1)
				cost_f[angle_idx] = float('-inf')
				idx_max = np.argmax(cost_f)
				angle_idx[i] = idx_max
				angle_est[i] = angle_grid[idx_max]
			
			if iter_num == iter_max:
				break
			cost = cost_f[idx_max]
			dif = np.abs(cost - cost_pre)/np.abs(cost_pre)
			angle_pre = angle_est.copy()
			cost_pre = cost
			iter_num += 1
		angle_est = np.sort(angle_est)
	a = steer_vecs(angle_est, L)
	S_est = np.linalg.inv(a.conj().T.dot(a)).dot(a.conj().T).dot(X)
	
	power_est = np.mean(np.abs(S_est)**2, 1)
	angle_est = angle_est/np.pi*180
	
	return angle_est, power_est

def log_likeli(E, L, K_cur, T):
	# function used in K_est_MDL
	diff = L - K_cur
	En = E[K_cur:]
	ld = T*np.log(np.prod(En)/(np.sum(En)/diff)**diff)
	return ld


def K_est_MDL(X, L_sub, K_max):
	# estimate the source numbers via MDL
	L, T = np.shape(X)
	R = space_smooth(X, L_sub, 1)
	T = T*(L - L_sub + 1)
	
	# eigenvalue decomposition
	vals, vecs = np.linalg.eigh(R)
	idx = np.argsort(-np.abs(vals))
	E = vals[idx]
	
	if K_max >= L_sub:
		K_max = L_sub - 1
	mdl_vec = np.zeros(K_max + 1)
	
	for ii in range(K_max + 1):
		mdl_vec[ii] = -log_likeli(E, L_sub, ii, T) + 0.5*(ii*(2*L_sub - ii) + 1)*np.log(T)
	
	idx = np.argmin(mdl_vec)
	K_est = idx
	
	return K_est, mdl_vec