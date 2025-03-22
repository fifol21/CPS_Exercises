import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.io import wavfile

# Wczytanie sygnału
file_name = "mowa.wav"
fs, x = wavfile.read(file_name)

plt.plot(x)
plt.title("Wczytany sygnał")
plt.show()
print("Oryginalny sygnał")
sd.play(x, fs)
sd.wait()

# DCT całego sygnału
c = dct(x, norm='ortho')
c = c / np.max(np.abs(c))
plt.stem(c)
plt.title("DCT - x")
plt.show()

# 25% współczynników
c_25 = np.zeros_like(c)
c_25[:len(c)//4] = c[:len(c)//4]
y_25 = idct(c_25, norm='ortho')
plt.stem(y_25)
plt.title("DCT - y_25")
plt.show()
print("25% współczynników")
sd.play(y_25, samplerate=fs)
sd.wait()

# 75% współczynników
c_75 = np.zeros_like(c)
c_75[-len(c)*3//4:] = c[-len(c)*3//4:]
y_75 = idct(c_75, norm='ortho')
plt.stem(y_75)
plt.title("DCT - y_75")
plt.show()
print("75% współczynników")
sd.play(y_75, samplerate=fs)
sd.wait()


t = np.arange(len(x)) / fs
amplitude = 0.5 * np.max(np.abs(x))
x_noisy = x + amplitude * np.sin(2 * np.pi * 250 * t)
x_noisy = x_noisy / np.max(np.abs(x_noisy))

plt.stem(x_noisy)
plt.title("Sygnał z zakłóceniem (250 Hz)")
plt.show()
print("Sygnał z zakłóceniem")
sd.play(x_noisy, samplerate=fs)
sd.wait()

c_noisy = dct(x_noisy, axis=0, norm='ortho')

# Indeks odpowiadający 250 Hz
freq_index = int(250 * len(c_noisy) / fs)
c_denoised = np.copy(c_noisy)
c_denoised[freq_index-15:freq_index+15] = 0


y_denoised = idct(c_denoised, axis=0, norm='ortho')

y_denoised = y_denoised / np.max(np.abs(y_denoised))

plt.stem(y_denoised)
plt.title("Sygnał po usunięciu zakłócenia")
plt.show()
print("Sygnał po filtracji")
sd.play(y_denoised, fs)
sd.wait()
