import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, decimate, firls

mat = sio.loadmat('lab08_fm.mat')
x = mat['x'].flatten()
fs = 2e6
fc = 200e3
t = np.arange(len(x)) / fs

# METODA 1 – IQ demodulacja

cos_fc = np.cos(2 * np.pi * fc * t)
sin_fc = np.sin(2 * np.pi * fc * t)

I = lfilter([1], [1], x * cos_fc)
Q = lfilter([1], [1], -x * sin_fc)

y = I + 1j * Q
demod1 = np.angle(y[1:] * np.conj(y[:-1])) / (2 * np.pi)
demod1 = decimate(demod1, int(fs // 8000))

# METODA 2 – filtr różniczkujący + BP + obwiednia
diff_filt = np.array([-1, 1])

# Filtr BP (80 dB tłumienia, FIR)
bp_filt = signal.firwin(numtaps=301, cutoff=[fc - 75e3, fc + 75e3], pass_zero=False, fs=fs, window=('kaiser', 8))


combined_filt = np.convolve(diff_filt, bp_filt)
x_filtered = lfilter(combined_filt, 1.0, x)


env = np.sqrt(lfilter([1], [1], x_filtered**2))

demod2 = decimate(env, int(fs // 8000))

# METODA 3 – BP + różniczka, FIR/IIR porównanie

bp_fir = signal.firwin(numtaps=301, cutoff=[fc - 75e3, fc + 75e3], pass_zero=False, fs=fs, window=('kaiser', 8))
x_bp_fir = lfilter(bp_fir, 1.0, x)
x_diff_fir = lfilter(diff_filt, 1.0, x_bp_fir)
demod3_fir = decimate(np.abs(x_diff_fir), int(fs // 8000))


b_iir, a_iir = signal.butter(4, [2*(fc - 75e3)/fs, 2*(fc + 75e3)/fs], btype='bandpass')
x_bp_iir = lfilter(b_iir, a_iir, x)
x_diff_iir = lfilter(diff_filt, 1.0, x_bp_iir)
demod3_iir = decimate(np.abs(x_diff_iir), int(fs // 8000))


bands = [0, fc - 100e3, fc - 75e3, fc + 75e3, fc + 100e3, fs/2]
bands = [b / (fs / 2) for b in bands]

desired = [0, 0, 1, 1, 0, 0]
numtaps = 301


bp_diff_fir = firls(numtaps, bands, desired)


x_opt = lfilter(bp_diff_fir, 1.0, x)
env_opt = np.sqrt(lfilter([1], [1], x_opt**2))
demod_opt = decimate(env_opt, int(fs // 8000))

plt.figure(figsize=(15, 8))
plt.subplot(3, 1, 1)
plt.title('Metoda 1: Demodulacja IQ')
plt.plot(demod1)
plt.subplot(3, 1, 2)
plt.title('Metoda 2: Różniczka + BP + obwiednia')
plt.plot(demod2)
plt.subplot(3, 1, 3)
plt.title('Metoda 3: BP + różniczka (FIR vs IIR)')
plt.plot(demod3_fir, label='FIR')
plt.plot(demod3_iir, label='IIR', alpha=0.7)
plt.figure(figsize=(10, 3))
plt.title('Metoda 4: Filtr różniczkujący BP (firls)')
plt.plot(demod_opt)
plt.tight_layout()
plt.show()
