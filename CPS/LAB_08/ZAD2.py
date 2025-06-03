import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Params
fs = 400_000  # docelowa częstotliwość próbkowania
fc1 = 100_000
fc2 = 110_000
modulation_depth = 0.25

# WŁASNY FILTR HILBERTA
def hilbert_filter(N=129):
    n = np.arange(-N//2, N//2 + 1)
    h = np.zeros_like(n, dtype=float)
    h[n != 0] = 2 / (np.pi * n[n != 0])
    h *= (1 - np.cos(2 * np.pi * n / N))  # Hanning window
    return h

def apply_hilbert(x):
    h = hilbert_filter()
    return signal.convolve(x, h, mode='same')

# Wczytanie i odwrócenie audio
fsx, x1 = wav.read("mowa8000.wav")
x1 = x1.astype(float) / np.max(np.abs(x1))
x2 = x1[::-1]

# Nadpróbkowanie
def upsample(x, fs_from, fs_to):
    return signal.resample_poly(x, fs_to, fs_from)

x1_up = upsample(x1, fsx, fs)
x2_up = upsample(x2, fsx, fs)

t = np.arange(len(x1_up)) / fs

# === MODULACJA DSB-C ===
x1_mod_dsb_c = (1 + modulation_depth * x1_up) * np.cos(2 * np.pi * fc1 * t)
x2_mod_dsb_c = (1 + modulation_depth * x2_up) * np.cos(2 * np.pi * fc2 * t)
y_dsb_c = x1_mod_dsb_c + x2_mod_dsb_c

# === MODULACJA DSB-SC ===
x1_mod_dsb_sc = modulation_depth * x1_up * np.cos(2 * np.pi * fc1 * t)
x2_mod_dsb_sc = modulation_depth * x2_up * np.cos(2 * np.pi * fc2 * t)
y_dsb_sc = x1_mod_dsb_sc + x2_mod_dsb_sc

# MODULACJA SSB-SC
x1_h = apply_hilbert(x1_up)
x2_h = apply_hilbert(x2_up)

# prawa wstęga (USB) dla stacji 1
x1_mod_ssb_usb = 0.5 * x1_up * np.cos(2 * np.pi * fc1 * t) - 0.5 * x1_h * np.sin(2 * np.pi * fc1 * t)

# lewa wstęga (LSB) dla stacji 2
x2_mod_ssb_lsb = 0.5 * x2_up * np.cos(2 * np.pi * fc2 * t) + 0.5 * x2_h * np.sin(2 * np.pi * fc2 * t)

y_ssb_sc = x1_mod_ssb_usb + x2_mod_ssb_lsb

# OPCJONALNA DEMODULACJA
def demodulate_am_dsb_c(y, fc):
    return signal.resample_poly(
        (2 * y * np.cos(2 * np.pi * fc * t)), fsx, fs)

def demodulate_am_dsb_sc(y, fc):
    return signal.resample_poly(
        (2 * y * np.cos(2 * np.pi * fc * t)), fsx, fs)

def demodulate_am_ssb(y, fc, side='+'):
    mixed_cos = y * 2 * np.cos(2 * np.pi * fc * t)
    mixed_sin = y * 2 * np.sin(2 * np.pi * fc * t)
    x = signal.resample_poly(mixed_cos, fsx, fs)
    xh = signal.resample_poly(mixed_sin, fsx, fs)
    return x - xh if side == '+' else x + xh

# DEMODULACJA DSB-C
x1_demod_dsb_c = demodulate_am_dsb_c(x1_mod_dsb_c, fc1)
x2_demod_dsb_c = demodulate_am_dsb_c(x2_mod_dsb_c, fc2)[::-1]

# DEMODULACJA DSB-SC
x1_demod_dsb_sc = demodulate_am_dsb_sc(x1_mod_dsb_sc, fc1)
x2_demod_dsb_sc = demodulate_am_dsb_sc(x2_mod_dsb_sc, fc2)[::-1]

# DEMODULACJA SSB-SC
x1_demod_ssb = demodulate_am_ssb(x1_mod_ssb_usb, fc1, side='+')
x2_demod_ssb = demodulate_am_ssb(x2_mod_ssb_lsb, fc2, side='-')[::-1]

# ZAPIS DEMODULACJI
def save_wav(filename, x, rate=8000):
    x = np.nan_to_num(x)  # convert NaN to 0
    x = x / np.max(np.abs(x) + 1e-12)  # avoid div by zero
    wav.write(filename, rate, (x * 32767).astype(np.int16))

save_wav("x1_dsb_c.wav", x1_demod_dsb_c)
save_wav("x2_dsb_c.wav", x2_demod_dsb_c)
save_wav("x1_dsb_sc.wav", x1_demod_dsb_sc)
save_wav("x2_dsb_sc.wav", x2_demod_dsb_sc)
save_wav("x1_ssb.wav", x1_demod_ssb)
save_wav("x2_ssb.wav", x2_demod_ssb)

# Można też zapisać sygnały radiowe:
save_wav("y_dsb_c.wav", y_dsb_c, fs)
save_wav("y_dsb_sc.wav", y_dsb_sc, fs)
save_wav("y_ssb_sc.wav", y_ssb_sc, fs)

def plot_freq_only(signals, fs, titles):
    plt.figure(figsize=(15, 4))

    for i, (signal, title) in enumerate(zip(signals, titles)):
        n = len(signal)
        freqs = np.fft.fftfreq(n, d=1/fs)
        spectrum = np.abs(np.fft.fft(signal))

        plt.subplot(1, len(signals), i + 1)
        plt.plot(freqs[:n//2], spectrum[:n//2])
        plt.title(f"{title} - Widmo")
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Amplituda")

    plt.tight_layout()
    plt.show()


plot_freq_only(
    [y_dsb_c, y_dsb_sc, y_ssb_sc],
    fs,
    ["DSB-C", "DSB-SC", "SSB-SC"]
)

