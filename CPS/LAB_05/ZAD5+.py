import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqs, lp2bp

f_center = 96e6
#TUTAJ BĘDZIE ALBO 2E6 (+1MHZ) LUB 2E4 (+1KHZ)
bw = 2e6  # szerokość pasma (±1 kHz)
w0 = 2 * np.pi * f_center # w srodkowa
B = 2 * np.pi * bw

N = 4
a_lp = np.poly([np.exp(1j * np.pi * (2 * k + N - 1) / (2 * N)) for k in range(N)]) # nasze zera
b_lp = [1]  # licznik = 1

#LP -> BP
b_bp, a_bp = lp2bp(b_lp, a_lp, wo=w0, bw=B) # zamiana LP na BP polega ona na zamianie s na s2+w2/Bs


w = np.linspace(2 * np.pi * 93e6, 2 * np.pi * 99e6, 2000)
w_Hz = w / (2 * np.pi)
_, h = freqs(b_bp, a_bp, w)
h_db = 20 * np.log10(np.abs(h) / np.max(np.abs(h)))


plt.figure(figsize=(10, 6))
plt.plot(w_Hz, h_db)
plt.axvline(f_center - bw, color='r', linestyle='--', label='Granice pasma zaporowego')
plt.axvline(f_center - bw / 2, color='g', linestyle='--', label='Granice pasma przepustowego')
plt.axvline(f_center + bw / 2, color='g', linestyle='--')
plt.axvline(f_center + bw, color='r', linestyle='--')
plt.axhline(-3, color='g', linestyle=':', label='-3 dB')
plt.axhline(-40, color='r', linestyle=':', label='-40 dB')
plt.title('Charakterystyka częstotliwościowa filtru BP (z LP ręcznie)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Wzmocnienie [dB]')
plt.grid(True)
plt.legend()

# TUTAJ TRZEBA SOBIE DOPASOWAĆ XLIM I YLIM W ZALEŻNOŚCI OD TEGO JAKIE CZĘSTOTLIWOSCI FILTUJEMY (CZY +1MHZ CZY +1kHZ)
plt.ylim(-60, 20)
plt.xlim(93e6, 100e6)
#plt.xlim(95.7e6, 96.3e6)
plt.show()
