import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, upfirdn, resample_poly, resample
import sounddevice as sd
import warnings
from scipy.io import wavfile
from scipy.interpolate import interp1d
import soxr

def manual_resample(x, fs_in, fs_out):
    gcd = np.gcd(fs_in, fs_out)
    up = fs_out // gcd
    down = fs_in // gcd #oblicznie wspolczynnikow decymacji(down) i interpolacji(up)

    numtaps = 161 #wplywa na szerokosc pasma przejsciowego
    cutoff = 0.5 / max(up, down)
    fir_filter = firwin(numtaps, cutoff, window='hamming')
    fir_filter /= np.sum(fir_filter)

    resampled = upfirdn(fir_filter, x, up=up, down=down) #reasampling dodaje zera pomiedzy probki filtruje fir a nastepnie usuwa probki

    expected_len = int(len(x) * fs_out / fs_in)
    resampled = resampled[:expected_len] #dopasowanie sygnalu

    return resampled

t = 1
fs_target = 48000

signals = [
    {'f': 1001.2, 'fs': 8000},
    {'f': 303.1,  'fs': 32000},
    {'f': 2110.4, 'fs': 48000}  #lista sygnałow
]

resampled_manual = []
resampled_poly = []

for s in signals:
    f = s['f']
    fs = s['fs']
    t_vec = np.arange(0, t, 1/fs) #tworztmy dwa puste pole, oblicz wektor czasu tvec z krokiem 1/f
    x = np.sin(2 * np.pi * f * t_vec)

    if fs != fs_target:
        x_manual = manual_resample(x, fs, fs_target)
        x_poly = resample_poly(x, fs_target, fs)
    else:
        x_manual = x[:int(fs_target * t)]
        x_poly = x_manual.copy()

    resampled_manual.append(x_manual)
    resampled_poly.append(x_poly)  #gdy czestotliwosc sygnalu roznci sie od docelowej to dokonuje reasamplingu

min_len = min(map(len, resampled_manual + resampled_poly))
resampled_manual = [x[:min_len] for x in resampled_manual] #przycina je do krotszej, aby mialy jednoakowe dlugosci
resampled_poly = [x[:min_len] for x in resampled_poly] #( po reasapmlinu moga miec rozne dlugosci)

x4_manual = sum(resampled_manual)
x4_poly = sum(resampled_poly)

t4 = np.arange(0, min_len) / fs_target
x4_expected = sum(np.sin(2 * np.pi * s['f'] * t4) for s in signals)

plt.figure(figsize=(12, 6))
plt.plot(t4, x4_expected, '--', label='x̄4 (analityczny)', alpha=0.8)
plt.plot(t4, x4_manual, label='x4 (manualny)', alpha=0.6)
plt.plot(t4, x4_poly, label='x4 (resample_poly)', alpha=0.6)
plt.xlim(0, 0.01)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.title('Porównanie trzech wersji sygnału x4')
plt.grid(True)
plt.tight_layout()
plt.show()


x4_poly_norm = x4_poly / np.max(np.abs(x4_poly))
print("Odtwarzanie sygnału z resample_poly...")
sd.play(x4_poly_norm, fs_target)
sd.wait()


# OPCJONALNIE
def mix_and_resample(file1, file2, target_sr, play=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr1, wav1 = wavfile.read('../CPS/LAB_08/x1.wav')
        sr2, wav2 = wavfile.read('../CPS/LAB_08/x2.wav')

    wav1_resampled = resample(wav1, int(len(wav1) * target_sr / sr1))
    if wav1_resampled.ndim > 1:
        wav1_resampled = wav1_resampled.mean(axis=1) #jesli wielokanalowy to konwersja do mono

    wav2_resampled = resample(wav2, int(len(wav2) * target_sr / sr2))
    if wav2_resampled.ndim > 1:
        wav2_resampled = wav2_resampled.mean(axis=1)

    print(f"Częstotliwość próbkowania po resamplingu: {target_sr} Hz")

    min_length = min(len(wav1_resampled), len(wav2_resampled))
    wav1_resampled = wav1_resampled[:min_length]  # przycina do takich samych dlugosci zeby mogly byc poprawmie zmiksowane
    wav2_resampled = wav2_resampled[:min_length]

    mixed_signal = wav1_resampled + wav2_resampled

    mixed_signal = mixed_signal / np.max(np.abs(mixed_signal)) #miesza i normalizyje zeby zapobiec przesterowaniu

    if play:
        print(f"Odtwarzanie zmiksowanego sygnału o częstotliowści {target_sr} Hz")
        sd.play(mixed_signal, target_sr)
        sd.wait()


mix_and_resample('x1.wav', 'x2.wav', 48000)
mix_and_resample('x1.wav', 'x2.wav', 44100)
print("------------------------------------------")


resampled_linear = []
resampled_sinc = []
resampled_soxr = []

for s in signals:
    f = s['f']
    fs = s['fs']
    t_vec = np.arange(0, t, 1/fs)
    x = np.sin(2 * np.pi * f * t_vec) #generuje sygnal sinus

    # --- Interpolacja liniowa ---
    t_target = np.arange(0, t, 1/fs_target)
    linear_interp = interp1d(t_vec, x, kind='linear', fill_value="extrapolate")
    x_linear = linear_interp(t_target)
    resampled_linear.append(x_linear)

    # --- Interpolacja sinc ---
    def sinc_interp(xn, tn, t_out):
        Ts = tn[1] - tn[0]
        y_out = np.zeros_like(t_out)

        for i, t in enumerate(t_out):
            sinc_values = np.sinc((t - tn) / Ts)
            y_out[i] = np.dot(sinc_values, xn) #dokladniejzza ale kosztowna obliczeniowo

        return y_out

    x_sinc = sinc_interp(x, t_vec, t_target)
    resampled_sinc.append(x_sinc)

    # --- SoXR Resampling (soxr-vhq) ---
    x_soxr = soxr.resample(x, fs, fs_target, quality='VHQ')
    resampled_soxr.append(x_soxr) #wyoskiej jakosci interpolacja uzywana w dokladnuch analizach audio


x4_linear = sum(resampled_linear)
x4_sinc = sum(resampled_sinc)
x4_soxr = sum(resampled_soxr)

t4 = np.arange(0, int(fs_target * t)) / fs_target
x4_expected = sum(np.sin(2 * np.pi * s['f'] * t4) for s in signals)

plt.figure(figsize=(14, 6))
plt.plot(t4, x4_expected, '--', label='x̄4 (analityczny)', alpha=0.7)
plt.plot(t4, x4_linear, label='x4 (interpolacja liniowa)', alpha=0.6)
plt.plot(t4, x4_sinc, label='x4 (rekonstrukcja sinc)', alpha=0.6)
plt.plot(t4, x4_soxr, label='x4 (soxr-vhq)', alpha=0.6)
plt.xlim(0, 0.01)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.title('Porównanie metod rekonstrukcji')
plt.grid(True)
plt.tight_layout()
plt.show()


print("Odtwarzanie x4 (liniowa)...")
sd.play(x4_linear / np.max(np.abs(x4_linear)), fs_target)
sd.wait()

print("Odtwarzanie x4 (sinc)...")
sd.play(x4_sinc / np.max(np.abs(x4_sinc)), fs_target)
sd.wait()

print("Odtwarzanie x4 (soxr)...")
sd.play(x4_soxr / np.max(np.abs(x4_soxr)), fs_target)
sd.wait()