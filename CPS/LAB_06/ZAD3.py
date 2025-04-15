import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, find_peaks
from scipy.signal import freqz


fs = 3200000     #  Częstotliwość próbkowania
N = int(32e6)      # liczba próbek
fc = -407812.5      #Częstotliwość przesunięcia stacji
bwSERV = 80000    # Pasmo jednej stacji FM
bwAUDIO = 16000   # Pasmo audio mono


with open("samples_100MHz_fs3200kHz.raw", "rb") as f:
    s = np.fromfile(f, dtype=np.uint8, count=2*N)

s = s.astype(np.int16) - 127

I = s[::2]
Q = s[1::2]
wideband_signal = I + 1j * Q # sygnal do postaci zespolonej, aby lepiej mozna bylo analizowa sygnal


t = np.arange(N) / fs
wideband_signal_shifted = wideband_signal * np.exp(-1j * 2 * np.pi * fc * t) # ustawiamy stacje na komkretna czestotliwosc


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='low')

b, a = butter_lowpass(80_000, fs, order=4)
wideband_signal_filtered = lfilter(b, a, wideband_signal_shifted)  # przepuszczamy tylko okolice +-80khz dla naszej stacji wybtanej


x = wideband_signal_filtered[::int(fs / (2 * bwSERV))]  # zmniejszamy liczbe probek bo nie potrzeba nam az tak szerokiego pasma

dx = x[1:] * np.conj(x[:-1])
y = np.angle(dx) # demodulacja doczytac jak to dziala

b2, a2 = butter_lowpass(16_000, 160_000, order=4)
y_filtered = lfilter(b2, a2, y)
ym = y_filtered[::int(160_000 / (2 * bwAUDIO))]

def deemphasis_filter(fs, tau=75e-6):
    dt = 1 / fs
    alpha = dt / (tau + dt)
    b = [alpha]
    a = [1, alpha - 1]
    return b, a

b_de, a_de = deemphasis_filter(32_000)
ym = lfilter(b_de, a_de, ym)

ym -= np.mean(ym)
ym /= 1.001 * np.max(np.abs(ym))

# plotting
f, Pxx = welch(wideband_signal, fs, nperseg=2048)
plt.figure()
plt.semilogy(f, Pxx)
plt.title("Widmo gęstości mocy (wideband_signal)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.grid()

threshold = 1e-3
peaks, _ = find_peaks(Pxx, height=threshold)
frequencies = f[peaks]
print("Częstotliwości stacji radiowych:", frequencies)


f, Pxx = welch(wideband_signal_shifted, fs, nperseg=2048)
plt.figure()
plt.semilogy(f, Pxx)
plt.title("Widmo gęstości mocy (wideband_signal_shifted)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.grid()


f, Pxx = welch(wideband_signal_filtered, fs, nperseg=2048)
plt.figure()
plt.semilogy(f, Pxx)
plt.title("Widmo gęstości mocy (wideband_signal_filtered)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.grid()


f, Pxx = welch(y_filtered, 160_000, nperseg=2048) # tutaj mozemy zmieniac y->y_filtered
plt.figure()
plt.semilogy(f, Pxx)
plt.title("Widmo gęstości mocy (y_filtered)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.grid()

# PSD for ym
f, Pxx = welch(ym, 32_000, nperseg=2048)
plt.figure()
plt.semilogy(f, Pxx)
plt.title("Widmo gęstości mocy (ym)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.grid()


plt.figure()
plt.specgram(wideband_signal_shifted, NFFT=1024, Fs=fs, noverlap=512, cmap="plasma")
plt.title("Spektrogram (wideband_signal_shifted)")
plt.xlabel("Czas [s]")
plt.ylabel("Częstotliwość [Hz]")
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.specgram(wideband_signal_filtered, NFFT=1024, Fs=fs, noverlap=512, cmap="plasma")
plt.title("Spektrogram (wideband_signal_filtered)")
plt.xlabel("Czas [s]")
plt.ylabel("Częstotliwość [Hz]")
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.specgram(y_filtered, NFFT=1024, Fs=160_000, noverlap=512, cmap="plasma")
plt.title("Spektrogram (y_filtered)")
plt.xlabel("Czas [s]")
plt.ylabel("Częstotliwość [Hz]")
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.specgram(ym, NFFT=1024, Fs=32_000, noverlap=512, cmap="plasma")
plt.title("Spektrogram sygnału audio (mono)")
plt.xlabel("Czas [s]")
plt.ylabel("Częstotliwość [Hz]")
plt.colorbar()
plt.tight_layout()
plt.show()





f_cutoff = 2100
fs_audio = 32_000
order = 1  #20 dB/dekadę


b_de2, a_de2 = butter(order, f_cutoff / (0.5 * fs_audio), btype='low')
ym2 = lfilter(b_de2, a_de2, ym)


b_pre, a_pre = butter(order, f_cutoff / (0.5 * fs_audio), btype='high')


w, h_de = freqz(b_de2, a_de2, worN=8000, fs=fs_audio)
_, h_pre = freqz(b_pre, a_pre, worN=8000, fs=fs_audio)
omega = 1e-17
plt.figure()
plt.semilogx(w, 20 * np.log10(abs(h_de) + omega), label="Filtr de-emfazy")
plt.semilogx(w, 20 * np.log10(abs(h_pre) +omega), label="Filtr pre-emfazy")
plt.axvline(f_cutoff, color='green', linestyle=':', label="2.1 kHz cutoff")
plt.title("Charakterystyki amplitudowo-częstotliwościowe")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Wzmocnienie [dB]")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()