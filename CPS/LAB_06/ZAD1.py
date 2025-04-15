import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import bilinear, lfilter, zpk2tf, freqz, TransferFunction

# Load data from butter.mat
mat_data = scipy.io.loadmat('butter.mat')
z, p, k = mat_data['z'].flatten(), mat_data['p'].flatten(), mat_data['k'].flatten()

# Parameters
fs = 16000
f_low, f_high = 1189, 1229

# Convert to transfer function (analog filter)
b_s, a_s = zpk2tf(z, p, k)

# Frequency response of the analog filter
w, h = freqz(b_s, a_s, worN=8000, fs=fs)

# Plot analog filter response
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), label='Analog H(s)')
plt.axvline(f_low, color='r', linestyle='--', label='f_low')
plt.axvline(f_high, color='g', linestyle='--', label='f_high')
plt.title('Characteristic amplitude-frequency of analog filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid()
plt.show()

# Convert analog filter to digital using bilinear transform
b_z, a_z = bilinear(b_s, a_s, fs)
w_z, h_z = freqz(b_z, a_z, worN=8000, fs=fs)

# Plot digital filter response
plt.figure()
plt.plot(w_z, 20 * np.log10(abs(h_z)), label='Digital H(z)')
plt.axvline(f_low, color='r', linestyle='--', label='f_low')
plt.axvline(f_high, color='g', linestyle='--', label='f_high')
plt.title('Amplitude-frequency characteristics of digital filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid()
plt.show()

# Generate test signal
t = np.arange(0, 1, 1/fs)
f1, f2 = 1209, 1272
signal = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

# Signal filtration (manual implementation)
filtered_signal = np.zeros_like(signal)
for n in range(len(signal)):
    for i in range(len(b_z)):
        if n - i >= 0:
            filtered_signal[n] += b_z[i] * signal[n - i]
    for j in range(1, len(a_z)):
        if n - j >= 0:
            filtered_signal[n] -= a_z[j] * filtered_signal[n - j]
    filtered_signal[n] /= a_z[0]

# Signal filtration using scipy
filtered_signal_lib = lfilter(b_z, a_z, signal)

# Comparison in time domain
plt.figure()
plt.plot(t, signal, label='Original signal')
plt.plot(t, filtered_signal, label='Filtered signal (manual)')
plt.plot(t, filtered_signal_lib, label='Filtered signal (scipy)', linestyle='--')
plt.title('Comparison of filtered and original signal in time domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# Comparison in frequency domain
fft_original = np.abs(np.fft.fft(signal))
fft_filtered = np.abs(np.fft.fft(filtered_signal))
fft_filtered_lib = np.abs(np.fft.fft(filtered_signal_lib))
frequencies = np.fft.fftfreq(len(signal), 1/fs)

plt.figure()
plt.plot(frequencies[:len(frequencies)//2], fft_original[:len(frequencies)//2], label='Original signal')
plt.plot(frequencies[:len(frequencies)//2], fft_filtered[:len(frequencies)//2], label='Filtered signal (manual)')
plt.plot(frequencies[:len(frequencies)//2], fft_filtered_lib[:len(frequencies)//2], label='Filtered signal (scipy)', linestyle='--')
plt.title('Comparison of filtered and original signal in frequency domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# Optional pre-warping
T = 1 / fs
omega_low = 2 * np.pi * f_low
omega_high = 2 * np.pi * f_high
omega_warped_low = 2 / T * np.tan(omega_low * T / 2)
omega_warped_high = 2 / T * np.tan(omega_high * T / 2)

# Rescale analog filter poles
p_warped = p * (omega_warped_low / omega_low)
b_s_warped, a_s_warped = zpk2tf(z, p_warped, k)

# Convert to digital filter after pre-warping
b_z_warped, a_z_warped = bilinear(b_s_warped, a_s_warped, fs)
w_z_warped, h_z_warped = freqz(b_z_warped, a_z_warped, worN=8000, fs=fs)

# Plot pre-warping results
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), label='Analog H(s)')
plt.plot(w_z, 20 * np.log10(abs(h_z)), label='Digital H(z)')
plt.plot(w_z_warped, 20 * np.log10(abs(h_z_warped)), label='Digital Hw(z) (pre-warping)')
plt.axvline(f_low, color='r', linestyle='--', label='f_low')
plt.axvline(f_high, color='g', linestyle='--', label='f_high')
plt.title('Amplitude-frequency characteristics with pre-warping')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid()
plt.show()