import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram, get_window

fs, data = wav.read('DontWorryBeHappy.wav')
data = data.astype(np.float32)
if data.ndim > 1:  # jeśli stereo
    data = data[:,0]

start = int(fs * 10)
end = start + int(fs * 3)  # wycinamy 3 sekundowe odcinki czasu zaczynajac od 10s
x = data[start:end]

def kodowanie_podpasmowe_python(x, M, q_bits):
                                        # M - liczba podpasm
                                        # q_bits- liczba bitów na każde podpasmo

    N = len(x)
    if isinstance(q_bits, int):
        q_bits = np.full(M, q_bits)   # jesli liczba tworzymy wektor dlugosci M
    elif len(q_bits) < M:
        q_bits = np.pad(q_bits, (0, M - len(q_bits)), 'edge')  # jesli lista za krotka dopelniamy wartoscia
    else:
        q_bits = q_bits[:M]   # jesli za dluga obcinamy


    window = get_window('hann', 2*M)
    step = M
    num_frames = (N - 2*M) // step + 1
    subbands = np.zeros((M, num_frames))

    # Analiza podpasmowa - bierzemy M "pasm" z krótkich FFT
    for i in range(num_frames):
        frame = x[i*step : i*step + 2*M] * window
        spectrum = np.fft.fft(frame)
        # Bierzemy tylko M pierwszych wartości - odpowiada to podpasmom
        subbands[:, i] = np.real(spectrum[:M])

    # Kwantyzacja w podpasmach
    yq = np.zeros_like(subbands)
    bits_used = 0
    for m in range(M):
        band = subbands[m, :]
        max_val = np.max(np.abs(band))
        if max_val == 0:
            yq[m, :] = band
            continue
        # Normalizacja do [-1,1]
        norm_band = band / max_val
        levels = 2**q_bits[m]
        quantized = np.round((norm_band + 1) / 2 * (levels - 1))
        quantized = np.clip(quantized, 0, levels - 1)
        # Rekonstrukcja
        yq[m, :] = ((quantized / (levels - 1)) * 2 - 1) * max_val

        bits_used += q_bits[m] * len(band)

    bps = bits_used / N  # średnia liczba bitów na próbkę

    # Synteza sygnału (odwrócona analiza)
    y = np.zeros(N)
    win_sum = np.zeros(N)

    for i in range(num_frames):
        frame_spectrum = np.zeros(2*M, dtype=complex)
        frame_spectrum[:M] = yq[:, i]
        frame_spectrum[M:] = np.conj(yq[::-1, i])
        frame_time = np.fft.ifft(frame_spectrum).real * window
        y[i*step:i*step+2*M] += frame_time
        win_sum[i*step:i*step+2*M] += window

    # Normalizacja przez sumę okien
    nonzero = win_sum > 0
    y[nonzero] /= win_sum[nonzero]

    return y, bps


# Warianty
variants = [
    (8, 6),
    (32, 6),
    (32, [8, 8, 7, 6, 4])  # rozszerzymy do 32 kanałów w kodowaniu_podpasmowe_python
]

results = []

for M, q in variants:
    y_rec, bps = kodowanie_podpasmowe_python(x, M, q)
    results.append((M, q, y_rec, bps))


def plot_spectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=512, noverlap=256)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylim([0, 8000])


for i, (M, q, y_rec, bps) in enumerate(results):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plot_spectrogram(x, fs, 'Oryginalny sygnał - spektrogram')
    plt.subplot(1, 2, 2)
    plot_spectrogram(y_rec, fs, f'Rekonstrukcja: {M} podpasm, q={q}, bps={bps:.2f}')
    plt.tight_layout()
    plt.show()


plt.plot(x, label='Oryginalny sygnał')
plt.title('PCM - Oryginalny sygnał')
plt.grid()
plt.legend()
plt.show()

original_bits_per_sample = 16

for M, q, y_rec, bps in results:
    compression_ratio = original_bits_per_sample / bps
    print(f'Wariant {M} podpasm, q={q} - bps={bps:.2f}, kompresja: {compression_ratio:.2f}x')