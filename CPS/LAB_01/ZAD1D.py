import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
#1
fs = 10000
T = 1
fn = 50
fm = 1
df = 5
t= np.arange(0, T, 1/fs)
#1
x_signal = np.sin(2*np.pi*fm*t)
frequency = fn + df * x_signal # f chwilowa zmianiajaca sie od xsignal
phase = 2*np.pi * np.cumsum(frequency)/fs # liczenie fazy
sfm_signal = np.sin(phase) # sygnał


plt.plot(t, x_signal, label='modulation signal')
plt.plot(t, sfm_signal, label='sfm signal')
plt.legend()
plt.show()

#2
fs_sampled = 25
t_sampled = np.arange(0, T, 1/fs_sampled)
sfm_signal_sampled = np.interp(t_sampled, t, sfm_signal)

plt.figure(figsize=(10, 9))
plt.scatter(t_sampled, sfm_signal_sampled,color="blue",  label='sfm signal sampled')
plt.xlim(0,0.2)
plt.plot(t,sfm_signal , color = "red", label='sfm signal', linewidth = 1)
plt.xlabel("Time (s)")
plt.grid()
plt.legend()
plt.show()

t_total = np.linspace(0, T, len(t_sampled), endpoint=False)
#3
def spectrum(signal,fs,title):  ## jak energia jest rozłozona w dziedzinie czestotliwosci
    f,Pxx = welch(signal,fs,nperseg=25)
    plt.semilogy(f,Pxx)
    plt.xlabel("freq[Hz]")
    plt.ylabel('spacial power density')
    plt.title(title)
    plt.grid()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
spectrum(sfm_signal,fs,"spectrum before sampling")
plt.subplot(1,2,2)
spectrum(sfm_signal_sampled,fs,"spectrum after sampling")
plt.show()





