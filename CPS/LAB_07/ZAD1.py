import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter, periodogram


fs = 1200
fc = 300
df = 200
f1, f2 = fc - df/2, fc + df/2
N_values = [128,129]


windows = ['boxcar', 'hann', 'hamming', 'blackman', 'blackmanharris']


for N in N_values:
    for win in windows:
        b = firwin(N, [f1, f2], pass_zero=False, fs=fs, window=win)
        w, h = freqz(b, worN=1024, fs=fs)
        amp_db = 20 * np.log10(np.abs(h) + 1e-8)
        phase = np.unwrap(np.angle(h))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(w, amp_db)
        plt.title(f'Amplituda - {win} for N = {N}')
        plt.grid(); plt.xlabel('Hz'); plt.ylabel('dB')
        plt.axvline(f1, color='r', linestyle='--')
        plt.axvline(f2, color='r', linestyle='--')

        plt.subplot(1, 2, 2)
        plt.plot(w, phase)
        plt.title(f'Faza - {win}')
        plt.grid(); plt.xlabel('Hz'); plt.ylabel('radiany')
        plt.tight_layout()
        plt.show()


t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*300*t) + np.sin(2*np.pi*500*t)


f, Pxx = periodogram(x, fs)
plt.semilogy(f, Pxx)
plt.title("Widmo sygnału przed filtracją")
plt.grid(); plt.xlabel("Hz")
plt.tight_layout(); plt.show()


for win in windows:
    b = firwin(129, [f1, f2], pass_zero=False, fs=fs, window=win)
    y = lfilter(b, 1.0, x)
    f, Pyy = periodogram(y, fs)

    plt.semilogy(f, Pyy)
    plt.title(f"Widmo po filtracji - {win}")
    plt.grid(); plt.xlabel("Hz")
    plt.tight_layout(); plt.show()
