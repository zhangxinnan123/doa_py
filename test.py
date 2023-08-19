
from doa_est_alg_fast import *
import numpy as np
from doa_est_alg import *
from doa_est_alg_weighted_spice_init import f_steer_vec
import matplotlib.pylab as plt

angle = np.array([-2.5, 2.5])
angle = np.deg2rad(angle)
L = 20
s = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(2,1)), dtype=complex)
snr = 10
A = f_steer_vec(angle, L)
noise = np.sqrt(1/np.power(10, snr/10))*(np.random.randn(L, 1) + 1j*np.random.randn(L, 1))
Y = A.dot(s) + noise

Y = Y.reshape((L, 1))
# p = gs_iaa(Y, 200, 30)
# p = fsiaa2(Y, 200, 5, 10)
# p = QN_PCG_IAA(Y, 200, 10, 15)
p = fun_qniaa(Y, 200, 10, 15)
theta_grid = np.linspace(-np.pi, np.pi, 200)
theta = np.arcsin(theta_grid/np.pi)*180/np.pi
p = 10*np.log10(np.fft.fftshift(p))
plt.plot(theta, p)

plt.show()
