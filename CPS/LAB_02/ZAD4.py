import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from CPS.LAB_02.ZAD1 import DCT_matrix

fs, samples = wav.read("mowa.wav")

# wyswietlanie nagrania
plt.plot(np.arange(len(samples)), samples)
plt.show()

M = 10
N = 256
fragments = []

samples_found= np.linspace(0,len(samples) - N,M,dtype=int)
for n1 in samples_found:
    n2 = n1 + N
    fragments.append(samples[n1:n2])

print(fragments)
plt.figure(figsize=(15,15))
for k in range(M):
    x_k = fragments[k]
    y_k = DCT_matrix(N) @ x_k
    freqs = np.linspace(0, fs, N)

    plt.subplot(M,2,2*k+1)
    plt.plot(np.arange(N), x_k)

    plt.subplot(M,2,2*k+2)
    plt.plot(freqs, y_k)

plt.show()



