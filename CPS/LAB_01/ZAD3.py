import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io

def cross_corr(x, y):

    len_x = len(x)
    len_y = len(y)
    corr = np.zeros(len_x - len_y + 1)
    for i in range(len_x - len_y + 1):
        corr[i] = np.sum(x[i:i+len_y] * y)
    return corr


mat_data = scipy.io.loadmat('adsl_x.mat')
signal = mat_data['x'].flatten()


M = 32
N = 512
block_size = M + N
K= len(signal)//block_size

for k in range(K):
    idx_start = k*block_size
    idx_end= k*block_size+M


    prefix=[idx_start,idx_end]


    correlation = np.correlate(signal, prefix, mode='valid')
    custom_correlation = cross_corr(signal, prefix)


    correlation /= np.max(correlation)
    custom_correlation /= np.max(custom_correlation)


    peaks, _ = sig.find_peaks(correlation, height=0.8)
    my_peaks, _ = sig.find_peaks(custom_correlation, height=0.8)


print("Pozycje powtórzeń prefiksu (pythona korelacja):", peaks)
print("Pozycje powtórzeń prefiksu (moja korelacja):", my_peaks)
plt.figure(figsize=(10, 5))
plt.plot(signal, label="Sygnał", alpha=0.7)
plt.plot(peaks, signal[peaks], 'ro', label="Szczyty korelacji")
plt.xlabel("Próbki")
plt.ylabel("Amplituda")
plt.legend()
plt.grid()
plt.show()
