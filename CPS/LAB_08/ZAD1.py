import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import firwin, lfilter, freqz, find_peaks
import scipy.io

fs = 1000 #probkowanie
fc = 200 # nosna

mat = scipy.io.loadmat('lab08_am.mat')
## print(mat.keys())
x= mat['s2'].squeeze() #bierzemy klucz odpowiedni do legitymacji , trzeba splascztc zeby ni ebylo np (1000,1)
N=len(x)
t=np.arange(N)/fs

# filtr hilberrta

num_taps = 1001
h = firwin(num_taps, cutoff = 0.95, window = 'hamming', pass_zero = False)

# transoformacja hilberta
ht_x = lfilter(h, 1.0, x)
delay = (num_taps - 1) // 2  #kompensujmy opoznienie pi/2 i przesuwamy zeby bylo zgodne z x i t
ht_x = np.roll(ht_x, -delay)

envelope = np.sqrt(x**2 + ht_x**2)

spectrum = np.fft.fft(envelope)
freqs = np.fft.fftfreq(N, 1/fs)
half = N//2
amps = 2 * np.abs(spectrum[:half])
freqs = freqs[:half]

plt.figure(figsize=(10, 4))
plt.plot(freqs, amps)
plt.title('Widmo obwiedni sygnału AM')
plt.xlabel('Hz')
plt.ylabel('Amp')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, x, label='x')
plt.plot(t, envelope, label='envelope', color='red')
plt.legend()
plt.xlim(0.2, 0.3)
plt.grid()
plt.title("Sygnał i jego obwiednia")
plt.show()

peaks, _ = find_peaks(amps, height=0.01)
huge = sorted(zip(amps[peaks], freqs[peaks]), reverse=True)[:3]
huge.sort(key=lambda x: x[1])

A1, f1 = huge[0]
A2, f2 = huge[1]
A3 , f3 = huge[2]

print(f": f1={f1:.3f} Hz, f2={f2:.3f} Hz, f3={f3:.3f} Hz")
print(f": A1={A1:.3f}, A2={A2:.3f}, A3={A3:.3f}")

m_t = 1 + A1 * np.cos(2 * np.pi * f1 * t) +A2 * np.cos(2 * np.pi * f2 * t) + A3 * np.cos(2 * np.pi * f3 * t)
x_reconstructed = (1+m_t) * np.cos(2 * np.pi * fc * t)

plt.figure(figsize=(10, 4))
plt.plot(t, x, label="Sygnał oryginalny", color='blue')
plt.plot(t, x_reconstructed, label="Sygnał zrekonstruowany", color='red')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

corr = np.corrcoef(x, x_reconstructed)[0, 1]
rel_error = np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)
print(f"Korelacja: {corr:.4f}")