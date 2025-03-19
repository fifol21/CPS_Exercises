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
#sd.play(x,fs)
#sd.wait()

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
#sd.play(y_25, samplerate=fs) #glosne bardzo ?
#sd.wait()

#75%

c_75 = np.zeros_like(c)
c_75[-len(c)*3//4:]=c[-len(c)*3//4:]
y_75 = idct(c_75, norm='ortho')
#sd.play(y_75,samplerate=fs)
#sd.wait()

#filtracja
c_filtrated = np.copy(c)
c_filtrated[np.abs(c_filtrated)<3000] = 0
y_filtrated = idct(c_filtrated, norm='ortho')
#sd.play(y_filtrated, fs)
#sd.wait()

t=np.arange(len(x))/fs
x_with_noise = x + 0.5* np.sin(2*np.pi*250*t)
x_with_noise = x_with_noise / np.max(np.abs(x_with_noise))
sd.play(x_with_noise, fs)
sd.wait()

plt.plot(x_with_noise)
plt.show()









