import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, buttord, freqs


def projektuj_filtr_analogowy_II(passband_hz, stopband_hz, Rp, Rs, R):
    # Konwersja częstotliwości na rad/s
    wp = np.array(passband_hz) * 2 * np.pi
    ws = np.array(stopband_hz) * 2 * np.pi

    _, wn = buttord(wp, ws, Rp, Rs, analog=True)
    b, a = butter(R, wn, btype='bandpass', analog=True)
    return b, a

def projektuj_filtr_analogowy(passband_hz, stopband_hz, Rp, Rs):
    # Konwersja częstotliwości na rad/s
    wp = np.array(passband_hz) * 2 * np.pi
    ws = np.array(stopband_hz) * 2 * np.pi

    order, wn = buttord(wp, ws, Rp, Rs, analog=True)
    print(f"Rząd filtru: {order}")

    b, a = butter(order, wn, btype='bandpass', analog=True)
    return b, a


def rysuj_charakterystyke(b, a, passband_hz, stopband_hz, tytul):
    f = np.logspace(np.log10(0.1 * passband_hz[0]), np.log10(10 * passband_hz[1]), 10000)
    w = 2 * np.pi * f
    _, h = freqs(b, a, w)
    h_db = 20 * np.log10(np.abs(h))

    plt.figure(figsize=(12, 6))
    plt.semilogx(f, h_db, label='Charakterystyka')
    plt.title(tytul)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Tłumienie [dB]')
    plt.grid(which='both', linestyle='--', alpha=0.7)

    plt.axvline(passband_hz[0], color='green', linestyle='--', label='Pasmo przepustowe')
    plt.axvline(passband_hz[1], color='green', linestyle='--')
    plt.axvline(stopband_hz[0], color='red', linestyle='--', label='Pasmo zaporowe')
    plt.axvline(stopband_hz[1], color='red', linestyle='--')
    plt.axhline(3, color='black', linestyle=':', label='3 dB')
    plt.axhline(-40, color='magenta', linestyle=':', label='-40 dB')

    plt.legend()
    plt.xlim(0.98 * passband_hz[0], 1.01 * passband_hz[1])
    plt.ylim(-150, 50)
    plt.show()


# Testowy filtr: 96 MHz ±1 MHz
Rp = 3  # Tłumienie w paśmie przepustowym [dB]
Rs = 40  # Tłumienie w paśmie zaporowym [dB]
passband_test = [95e6, 97e6]
stopband_test = [94e6, 98e6]  # Założone pasmo zaporowe

b_test, a_test = projektuj_filtr_analogowy(passband_test, stopband_test, Rp, Rs)
rysuj_charakterystyke(b_test, a_test, passband_test, stopband_test, 'Filtr testowy (96 MHz ±1 MHz)')

# Docelowy filtr: 96 MHz ±100 kHz
passband_docelowy = [95.9e6, 96.1e6]
stopband_docelowy = [95.8e6, 96.2e6]  # Wąskie pasmo zaporowe
R = 4

b_docelowy, a_docelowy = projektuj_filtr_analogowy_II(passband_docelowy, stopband_docelowy, Rp, Rs, R)
rysuj_charakterystyke(b_docelowy, a_docelowy, passband_docelowy, stopband_docelowy, 'Filtr docelowy (96 MHz ±100 kHz)')