import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


fs = 256e3
f3dB = 64e3
f_stop = fs / 2
A_stop = 40

filter_types = ['butter', 'cheby1', 'cheby2', 'ellip']
filters = {}

for f_type in filter_types:
    if f_type == 'butter':
        N, Wn = signal.buttord(f3dB, f_stop, 3, A_stop, analog=True)
        b, a = signal.butter(N, Wn, btype='low', analog=True)
    elif f_type == 'cheby1':
        N, Wn = signal.cheb1ord(f3dB, f_stop, 3, A_stop, analog=True)
        b, a = signal.cheby1(N, 3, Wn, btype='low', analog=True)
    elif f_type == 'cheby2':
        N, Wn = signal.cheb2ord(f3dB, f_stop, 3, A_stop, analog=True)
        b, a = signal.cheby2(N, A_stop, Wn, btype='low', analog=True)
    elif f_type == 'ellip':
        N, Wn = signal.ellipord(f3dB, f_stop, 3, A_stop, analog=True)
        b, a = signal.ellip(N, 3, A_stop, Wn, btype='low', analog=True)

    filters[f_type] = (b, a, N)


plt.figure(figsize=(10, 6))
for f_type, (b, a, N) in filters.items():
    w, h = signal.freqs(b, a, worN=np.logspace(3, 6, 1000))
    plt.semilogx(w / (2 * np.pi), 20 * np.log10(abs(h)), label=f'{f_type} (N={N})')

plt.axvline(f3dB, color='k', linestyle='--', label='64 kHz (f3dB)')
plt.axvline(f_stop, color='r', linestyle='--', label='128 kHz (fs/2)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Wzmocnienie [dB]')
plt.title('Charakterystyki amplitudowe filtrów')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()


plt.figure(figsize=(10, 6))
for f_type, (b, a, N) in filters.items():
    poles = np.roots(a)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', label=f'{f_type} (N={N})')

plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.xlabel('Część rzeczywista')
plt.ylabel('Część urojona')
plt.title('Rozkład biegunów filtrów')
plt.legend()
plt.grid()
plt.show()
