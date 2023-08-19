
from scipy.linalg import toeplitz, solve_toeplitz
import numpy as np
from tool import *

def GS_IAA(X, G, Maxiter):
    L, T = np.shape(X)
    
    s_hat = np.fft.fft(X, G, 0)/L
    p_hat = np.squeeze(np.square(np.abs(s_hat)))

    for iter in range(Maxiter):
        p_hat_pre = p_hat
        R = com_R(p_hat, L, G)

        r = R[1:, 0]

        ww = np.append(1, solve_toeplitz(R[:-1, 0], -r))
        ws = np.append(0, np.flip(ww[1:])).conj()
        alpha = np.real(R[0, 0] + r.conj().T.dot(ww[1:]))

        tt = np.flip(ww) * np.array(range(1, L+1))
        ts = np.flip(ws) * np.array(range(1, L+1))

        tmp1 = comT(tt, np.append(tt[0], np.zeros(len(ww)-1)), ww.conj())
        tmp2 = comT(ts, np.append(ts[0], np.zeros(len(ws)-1)), ws.conj())

        cb = (tmp1 - tmp2) / alpha
        c = np.concatenate((np.flip(cb.conj()), np.zeros(G-2*L+1), cb[:-1]))


        cf = np.fft.fft(c, G)
        b = np.append(cf[0], np.flip(cf[1:]))

        w = p_hat * np.square(np.real(b))
        z3 = comRix(ww, ws, alpha, X)

        beta = p_hat * np.fft.fft(z3, G, 0)
        p_hat = np.abs(beta) / np.sqrt(w)
        dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
        # print(dif)

    return p_hat



def FSIAA1(X, G, Ns, Maxiter):
    L, T = np.shape(X)
    l_n = int(L/Ns)

    s_hat = np.fft.fft(X, G, 0)/L
    p_init = np.squeeze(np.square(np.abs(s_hat)))
    phi = np.zeros(G)
    for l in range(l_n):
        p_hat = p_init
        for iter in range(Maxiter):

            R = com_R(p_hat, Ns, G)

            r = R[1:, 0]

            ww = np.append(1, solve_toeplitz(R[:-1, 0], -r))
            ws = np.append(0, np.flip(ww[1:])).conj()
            alpha = np.real(R[0, 0] + r.conj().T.dot(ww[1:]))

            tt = np.flip(ww) * np.array(range(1, Ns+1))
            ts = np.flip(ws) * np.array(range(1, Ns+1))

            tmp1 = comT(tt, np.append(tt[0], np.zeros(len(ww)-1)), ww.conj())
            tmp2 = comT(ts, np.append(ts[0], np.zeros(len(ws)-1)), ws.conj())

            cb = (tmp1 - tmp2) / alpha
            c = np.concatenate((np.flip(cb.conj()), np.zeros(G-2*Ns+1), cb[:-1]))


            cf = np.fft.fft(c, G)
            b = np.append(cf[0], np.flip(cf[1:]))

            w = p_hat * np.square(np.real(b))

            z3 = comRix(ww, ws, alpha, X[l*Ns: (l+1)*Ns])

            beta = p_hat * np.fft.fft(z3, G, 0) * np.exp(-1j * 2 * np.pi * l * Ns * np.array(range(G))/G)
            p_hat = np.abs(beta) / np.sqrt(w)

        phi += np.real(p_hat)
    return phi/L
        # print(dif)

def FSIAA2(X, G, Ns, Maxiter):
    L, T = np.shape(X)
    l_n = int(L/Ns)

    s_hat = np.fft.fft(X, G, 0)/L
    p_init = np.squeeze(np.square(np.abs(s_hat)))

    p_hat = p_init
    for iter in range(Maxiter):

        R = com_R(p_hat, Ns, G)

        r = R[1:, 0]

        ww = np.append(1, solve_toeplitz(R[:-1, 0], -r))
        ws = np.append(0, np.flip(ww[1:])).conj()
        alpha = np.real(R[0, 0] + r.conj().T.dot(ww[1:]))

        tt = np.flip(ww) * np.array(range(1, Ns+1))
        ts = np.flip(ws) * np.array(range(1, Ns+1))

        tmp1 = comT(tt, np.append(tt[0], np.zeros(len(ww)-1)), ww.conj())
        tmp2 = comT(ts, np.append(ts[0], np.zeros(len(ws)-1)), ws.conj())

        cb = (tmp1 - tmp2) / alpha
        c = np.concatenate((np.flip(cb.conj()), np.zeros(G-2*Ns+1), cb[:-1]))


        cf = np.fft.fft(c, G)
        b = np.append(cf[0], np.flip(cf[1:]))

        w = p_hat * np.square(np.real(b))
        phi = np.zeros(G)
        for l in range(l_n):
            z3 = comRix(ww, ws, alpha, X[l*Ns: (l+1)*Ns])

            beta = p_hat * np.fft.fft(z3, G, 0) * np.exp(-1j * 2 * np.pi * l * Ns * np.array(range(G))/G)
            p_hat = np.abs(beta) / np.sqrt(w)
            phi += p_hat
        p_hat = phi/L


    return phi/L


def QN_PCG_IAA(X, G, K, Maxiter):
    L, T = np.shape(X)
    
    s_hat = np.fft.fft(X, G, 0)/L
    p_hat = np.squeeze(np.square(np.abs(s_hat)))

    for iter in range(Maxiter):
        p_hat_pre = p_hat
        R = com_R(p_hat, L, G)

        r_n = R[1:, 0]
        r_sub = R[: -1, 0]
        r = R[1: K, 0]
        ww_k = np.append(1, solve_toeplitz(R[:K-1, 0], -R[1: K, 0]))
        a = ww_k[1:]
        ws_k = np.append(0, np.flip(a)).conj()
        alpha = R[0, 0] + r.conj().T.dot(a)

        a_mb = np.append(1, a) / np.sqrt(alpha)
        a_nb = np.append(a_mb, np.zeros(L-K-1))

        # QNPCG
        a_n = qnpcg_fast(r_sub, -r_n, K=K, ww=ww_k, ws=ws_k, alpha=alpha, a_nb=a_nb, x=None, err=1e-5)

        ww = np.append(1, a_n)
        ws = np.append(0, np.flip(a_n).conj())
        alpha = R[0, 0] + r_n.conj().T.dot(a_n)



        tt = np.flip(ww) * np.array(range(1, L+1))
        ts = np.flip(ws) * np.array(range(1, L+1))

        tmp1 = comT(tt, np.append(tt[0], np.zeros(len(ww)-1)), ww.conj())
        tmp2 = comT(ts, np.append(ts[0], np.zeros(len(ws)-1)), ws.conj())

        cb = (tmp1 - tmp2) / alpha
        c = np.concatenate((np.flip(cb.conj()), np.zeros(G-2*L+1), cb[:-1]))


        cf = np.fft.fft(c, G)
        b = np.append(cf[0], np.flip(cf[1:]))

        w = p_hat * np.square(np.real(b))
        z3 = comRix(ww, ws, alpha, X)

        beta = p_hat * np.fft.fft(z3, G, 0)
        p_hat = np.abs(beta) / np.sqrt(w)
        # dif = np.linalg.norm(p_hat - p_hat_pre)/np.linalg.norm(p_hat_pre)
    return p_hat

def QN_IAA(X, G, K, Maxiter):
    L, T = np.shape(X)
    
    s_hat = np.fft.fft(X, G, 0)/L
    p_hat = np.squeeze(np.square(np.abs(s_hat)))

    for iter in range(Maxiter):

        R = com_R(p_hat, L, G)

        r = R[1: K, 0]
        ww_k = np.append(1, solve_toeplitz(R[:K-1, 0], -R[1: K, 0]))
        a = ww_k[1:]
        ws_k = np.append(0, np.flip(a)).conj()
        alpha = R[0, 0] + r.conj().T.dot(a)

        a_mb = np.append(1, a) / np.sqrt(alpha)
        a_nb = np.append(a_mb, np.zeros(L-K))


        tt = np.flip(ww_k) * np.array(range(1, K+1))
        ts = np.flip(ws_k) * np.array(range(1, K+1))

        tmp1 = comT(tt, np.append(tt[0], np.zeros(len(ww_k)-1)), ww_k.conj())
        tmp2 = comT(ts, np.append(ts[0], np.zeros(len(ws_k)-1)), ws_k.conj())

        cb = (tmp1 - tmp2) / alpha
        c = np.concatenate((np.flip(cb.conj()), np.zeros(G-2*K+1), cb[:-1]))


        cf = np.fft.fft(c, G)
        b = np.append(cf[0], np.flip(cf[1:]))

        phi = b + (L-K)*np.square(np.abs(np.fft.fft(a_mb, G, 0)))

        aax = comAAx(a_nb, X, L-K+1)
        q_x = np.concatenate((np.zeros(L-K), comRix(ww_k, ws_k, alpha, X[L-K: ]))) + aax

        s_hat = np.fft.fft(q_x, G, 0) / phi
        p_hat = np.square(np.abs(s_hat))

    return p_hat

def RC_IAA(X,)