import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import scipy.signal as signal

y1, fs1 = librosa.load("car.wav", sr=None)
y2, fs2 = librosa.load("bird.wav", sr=None)

min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]
y_total = y1 + y2

#FFT
N_total = len(y_total)
yf_total = np.fft.fft(y_total)
xf_total = np.fft.fftfreq(N_total, 1/fs2)

plt.plot(xf_total[:N_total//2], np.abs(yf_total[:N_total//2]))
plt.title('Widmo FFT suma')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid()
plt.show()

N1 = len(y1)
yf1 = np.fft.fft(y1)
xf1 = np.fft.fftfreq(N1, 1/fs1)

plt.plot(xf1[:N1//2], np.abs(yf1[:N1//2]))
plt.title('Widmo FFT - silnik')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid()
plt.show()

N2 = len(y2)
yf2 = np.fft.fft(y2)
xf2 = np.fft.fftfreq(N2, 1/fs2)

plt.plot(xf2[:N2//2], np.abs(yf2[:N2//2]))
plt.title('Widmo FFT - ptak')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid()
plt.show()

#spektogram
D_total = librosa.stft(y_total)
S_db = librosa.amplitude_to_db(np.abs(D_total), ref=np.max)
librosa.display.specshow(S_db, sr=fs1, x_axis='time', y_axis='hz')
plt.colorbar(format='%+1.5f dB')
plt.title('Spektrogram - suma')
plt.show()


D_1 = librosa.stft(y1)
S_db_1 = librosa.amplitude_to_db(np.abs(D_1), ref=np.max)
librosa.display.specshow(S_db_1, sr=fs1, x_axis='time', y_axis='hz')
plt.colorbar(format='%+1.5f dB')
plt.title('Spektrogram - silnik')
plt.show()


D_2 = librosa.stft(y2)
S_db_2 = librosa.amplitude_to_db(np.abs(D_2), ref=np.max)
librosa.display.specshow(S_db_2, sr=fs2, x_axis='time', y_axis='hz')
plt.colorbar(format='%+1.5f dB')
plt.title('Spektrogram - ptak')
plt.show()


cutoff = 1500
order = 4
b, a = signal.butter(order, cutoff / (0.5 * fs1), btype='low')


y_filtered = signal.lfilter(b, a, y_total)

N_filtered = len(y_filtered)
yf_filtered = np.fft.fft(y_filtered)
xf_filtered = np.fft.fftfreq(N_filtered, 1/fs1)


D_filtered = librosa.stft(y_filtered)
S_filtered = librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max)
librosa.display.specshow(S_filtered, sr=fs1, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spektrogram - po filtrze")
plt.show()

zeros = np.roots(b)
poles = np.roots(b)

plt.plot(np.real(zeros), np.imag(zeros), 'go', label='zeros')
plt.plot(np.real(poles), np.imag(poles), 'k*', label='bieguny')
plt.legend()
plt.show()