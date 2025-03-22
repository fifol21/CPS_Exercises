import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.io import wavfile

#wczytanie sygnalu

file_name="mowa.wav"
fs ,x = wavfile.read(file_name)

plt.plot(x)
plt.title("Wczytany sygnał")
plt.show()
print("oryginal")
sd.play(x,fs)
sd.wait()

# DCT calego sygnalu
c=dct(x, norm='ortho')
c = c / np.max(np.abs(c))
plt.stem(c)
plt.title("DCT - x")
plt.show()

# 25%
c_25 = np.zeros_like(c)
c_25[:len(c)//4]=c[:len(c)//4]
y_25 = idct(c_25, norm='ortho')
plt.stem(y_25)
plt.title("DCT - y_25")
plt.show()
print("25%")
sd.play(y_25, samplerate=fs)
sd.wait()

#75%

c_75 = np.zeros_like(c)
c_75[-len(c)*3//4:]=c[-len(c)*3//4:]
y_75 = idct(c_75, norm='ortho')
plt.stem(y_75)
print("75%")
plt.title("DCT - y_75")
plt.show()
sd.play(y_75,samplerate=fs)
sd.wait()


t=np.arange(len(x))/fs
x = x + 0.5 * np.sin(2*np.pi*250*t)
x = x/ np.max(np.abs(x))
plt.stem(x)
plt.show()
print("z szumem")
sd.play(x, fs)
sd.wait()