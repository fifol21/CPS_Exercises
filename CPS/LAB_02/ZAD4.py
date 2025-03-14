
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import ZAD1
from CPS.LAB_02.ZAD1 import DCT_matrix


f, samples = wavfile.read("mowa.wav")


plt.plot(np.arange(len(samples)), samples)

plt.title("Waveform of Audio Signal")
plt.show()

M = 10
N = 256
fragments=[]
A = DCT_matrix(N)

begin_ = np.linspace(0,len(samples)-N,M,dtype=int)
for n1 in begin_:
    n2 = n1 + N
    fragments.append(samples[n1:n2])


plt.figure(figsize=(20,2*M))
for k in range(M):
    x_k = fragments[k]
    y_k = A @ x_k
    freqs = np.linspace(0, f / 2, N)

    plt.subplot(M,2 ,2*k+1)
    plt.plot(np.arange(N),x_k)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")


    plt.subplot(M,2 ,2*k+2)
    plt.plot(freqs,y_k)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")



plt.show()