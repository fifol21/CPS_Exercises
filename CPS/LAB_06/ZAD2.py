import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bilinear, lfilter, zpk2tf, freqz
from scipy.io import wavfile
from scipy.signal import spectrogram
import warnings

# Wczytanie danych z pliku .mat
mat_data = scipy.io.loadmat('butter.mat')
z, p, k = mat_data['z'].flatten(), mat_data['p'].flatten(), mat_data['k'].flatten()

fs = 16000
f_low, f_high = 1189, 1229

# Konwersja do funkcji przenoszenia (filtr analogowy)
b_s, a_s = zpk2tf(z, p, k)

# Odpowiedź częstotliwościowa filtru analogowego
w, h = freqz(b_s, a_s, worN=8000, fs=fs)

# Przekształcenie filtru analogowego do cyfrowego za pomocą transformacji biliniowej
b_z, a_z = bilinear(b_s, a_s, fs)
w_z, h_z = freqz(b_z, a_z, worN=8000, fs=fs)

# Wczytanie pliku audio
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fs, sX = wavfile.read('../LAB_07/s7.wav')

# Obliczenie spektrogramu sygnału przed filtracją
f, t, Sxx = spectrogram(sX, fs=fs, nperseg=4096, noverlap=4096-512, scaling='density')

# Wizualizacja spektrogramu przed filtracją
plt.figure(figsize=(12, 5))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Częstotliwość [Hz]')
plt.xlabel('Czas [s]')
plt.title('Spektrogram sygnału sX przed filtracją')
plt.colorbar(label='Amplituda [dB]')
plt.ylim(0, 2000)  # Tylko zakres DTMF
plt.grid()
plt.show()

# Filtracja sygnału sX (implementacja filtru BP)
filtered_sX = np.zeros_like(sX)
for n in range(len(sX)):
    for i in range(len(b_z)):
        if n - i >= 0:
            filtered_sX[n] += b_z[i] * sX[n - i]
    for j in range(1, len(a_z)):
        if n - j >= 0:
            filtered_sX[n] -= a_z[j] * filtered_sX[n - j]
    filtered_sX[n] /= a_z[0]

# Skompensowanie opóźnienia sygnału (przesunięcie o połowę długości filtra)
delay = len(b_z) // 2
filtered_sX_compensated = np.roll(filtered_sX, -delay)
filtered_sX_lib = lfilter(b_z, a_z, sX)

time = np.arange(len(sX)) / fs
time_filt = np.arange(len(filtered_sX_compensated)) / fs



time_filt_lib = np.arange(len(filtered_sX_lib)) / fs
plt.figure(figsize=(12, 5))
plt.plot(time, sX, label='Oryginalny sygnał')
plt.plot(time_filt_lib, filtered_sX_lib, label='Po filtracji')
plt.title('Sygnał przed i po filtracji w dziedzinie czasu')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.show()


## ZADANIE 2.1

#Algorytm Goertzla
import numpy as np

# Funkcja Goertzla
def goertzel(samples, fs, target_freq):
    N = len(samples)
    k = int(0.5 + N * target_freq / fs)
    w = 2 * np.pi * k / N
    coeff = 2 * np.cos(w)

    s0, s1, s2 = 0.0, 0.0, 0.0
    for n in range(N):
        s0 = samples[n] + coeff * s1 - s2
        s2 = s1
        s1 = s0

    real = s1 - s2 * np.cos(w)
    imag = s2 * np.sin(w)
    return real**2 + imag**2

# Częstotliwości DTMF
dtmf_low = [697, 770, 852, 941]
dtmf_high = [1209, 1336, 1477]

# Mapa klawiszy
dtmf_keys = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#'
}

# Parametry
segment_duration = 0.5  # sekundy
N = int(segment_duration * fs)
num_segments = int(len(sX) / N)
threshold = 1e6  # Próg mocy dla detekcji tonu – można zmienić

# Analiza segmentów
# Wybrane czasy segmentów (w sekundach)
selected_times = [0.5, 1.5, 2.5, 4.5, 5.5]

for t in selected_times:
    i = int(t / segment_duration)
    segment_start_time = i * segment_duration
    segment = sX[i*N:(i+1)*N]

    powers = {}
    print(f"\n--- Czas {segment_start_time:.1f}s: Amplitudy (moc Goertzla) ---")
    for f in dtmf_low + dtmf_high:
        powers[f] = goertzel(segment, fs, f)
        print(f"{f} Hz: {powers[f]:.2e}")

    max_power = max(powers.values())
    if max_power < threshold:
        print(f"Brak wyraźnego tonu (max moc: {max_power:.2e})")
        continue

    # Znajdź dominujące częstotliwości
    low_freq = max(dtmf_low, key=lambda f: powers[f])
    high_freq = max(dtmf_high, key=lambda f: powers[f])

    key = dtmf_keys.get((low_freq, high_freq), '?')
    print(f"==> Naciśnięto '{key}' (low: {low_freq} Hz, high: {high_freq} Hz)")



## ZADANIE 2.2

def resonator_filter(freq, fs, r=0.99):
    w = 2 * np.pi * freq / fs
    b = [1 - r]
    a = [1, -2 * r * np.cos(w), r ** 2]
    return b, a

dtmf_freqs = dtmf_low + dtmf_high

for t in selected_times:
    i = int(t / segment_duration)
    segment = sX[i*N:(i+1)*N]

    energies = {}

    print(f"\n--- Czas {t:.1f}s: Energia sygnału po filtrach IIR ---")
    for freq in dtmf_freqs:
        b, a = resonator_filter(freq, fs, r=0.99)
        filtered = lfilter(b, a, segment)
        energy = np.sum(filtered**2)
        energies[freq] = energy
        print(f"{freq} Hz: {energy:.2e}")

    low_freq = max(dtmf_low, key=lambda f: energies[f])
    high_freq = max(dtmf_high, key=lambda f: energies[f])

    key = dtmf_keys.get((low_freq, high_freq), '?')
    print(f"==> Naciśnięto '{key}' (low: {low_freq} Hz, high: {high_freq} Hz)")


## ZADANIE 2.3 (OPCJONALNIE)
#ten algorytm decyzyjny jest w zadaniu 2.1 i 2.2 jak wybiera te maksymalne częstotliwości i tam wyznacza wystukane cyfry
