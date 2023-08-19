from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from scipy.linalg import toeplitz

def com_R(p, N, K):
    '''
    Compute R using FFT
    R = A*diag(p)*A'
    '''
    x = np.fft.fft(p, K, 0)
    r = x[:N]
    R = toeplitz(r.conj())
    return R

def comT(c, r, x):
    '''
    Compute y = A*x
    A = toeplitz(c, r)
    '''
    N = len(r)
    z = np.append(r, 0)
    z = np.append(z, np.flip(c[1:]))
    lam = np.fft.fft(z)
    tmp = np.fft.ifft(np.append(x, np.zeros(N)))
    ytmp = np.fft.fft(lam * tmp)
    y = ytmp[:N]

    return y


def comRix(ww, ws, alpha, x):
    '''
    Compute y = inv(R)*x
    '''
    z1 = comT(np.append(ww[0], np.zeros(len(ww)-1)), ww.conj(), x)
    z2 = comT(ww, np.append(ww[0], np.zeros(len(ww)-1)), z1)
    z3 = comT(np.append(ws[0], np.zeros(len(ww)-1)), ws.conj(), x)
    z4 = comT(ws, np.append(ws[0], np.zeros(len(ww)-1)), z3)
    y = (z2 - z4) / alpha

    return y

def comAAx(a, x, K):
    '''
    y = A * A' * x
    A = toeplitz(a, zeros(N-K+1))
    '''
    N = len(a)
    c = np.append(a[0], np.zeros(N-1))
    x2 = comT(c, a.conj(), x)
    x2 = x2[: K]
    y = comT(a, c, np.concatenate((x2, np.zeros(N-K))))
    return y

def qnpcg_fast(r, b, K, ww, ws, alpha, a_nb, x=None, err=1e-3, max_iter=512):
    '''
    A = toeplitz(r)
    '''
    if x is None:
        x = np.zeros_like(b)
    N = len(b)+1
    e = b - comT(r, r.conj(), x)
    # z = Q*e

    z = np.concatenate((np.zeros(N-1-K), comRix(ww, ws, alpha, e[N-K-1:]))) + comAAx(a_nb, e, N-K)
    e_old = e
    z_old = z
    p = z
    w = comT(r, r.conj(), p)
    alpha = e.conj().T.dot(z) / (p.conj().T.dot(w))
    x = x + alpha * p
    e = e - alpha * w
    pho = np.sum(e * e.conj())
    for iter in range(max_iter):
        z = np.concatenate((np.zeros(N-1-K), comRix(ww, ws, alpha, e[N-K-1:]))) + comAAx(a_nb, e, N-K)
        beta = e.conj().T.dot(z) / e_old.conj().T.dot(z_old)
        p = z + beta * p
        w = comT(r, r.conj(), p)
        alpha = e.conj().T.dot(z) / (p.conj().T.dot(w))
        x = x + alpha * p
        e_old = e
        
        e = e - alpha * w
        z_old = z
        pho = np.sum(e*e.conj())
        if pho < err:
            break
    return x



    