import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def lab11_kwant(d, levels=16):
    d_min = np.min(d)
    d_max = np.max(d)
    step = (d_max - d_min) / (levels - 1)
    indices = np.round((d - d_min) / step) # do poziomow
    dq = indices * step + d_min
    dq = np.clip(dq, d_min, d_max) # dla bezpieczenstwa zeby nie wyszlo poza zakres
    return dq

Fs, x = wavfile.read('DontWorryBeHappy.wav')
x = x.astype(np.float64)

if len(x.shape) > 1 and x.shape[1] > 1: #-> przejscie na mono
    x = np.mean(x, axis=1)

a = 0.9545

# KODER
x_shifted = np.insert(x[:-1], 0, 0)  # tworzymy przesunieta  wersje sygnalu - aby miec dostep do poprzednia probka
d = x - a * x_shifted #- roznica pomiedzy poprzednia a przewidywana - to jest major think w DCPM bledy predyckji to wlasnie to przesyla DCPM ( o ile pomylilismy sie )
dq = lab11_kwant(d, levels=128)

# --- DEKODER ---
y = np.zeros_like(x)
for n in range(1, len(x)):
    y[n] = dq[n] + a * y[n - 1] # odtwarzamy sygnal na podstawie roznicy i poprzedniej wartosci

# --- PORÓWNANIE ---
n = np.arange(len(x))
plt.figure(figsize=(12, 5))
plt.plot(n, x, label='Oryginalny sygnał x(n)', color='blue')
plt.plot(n, y, label='Zrekonstruowany sygnał y(n)', color='red', alpha=0.6)
plt.title('Porównanie sygnału oryginalnego i zrekonstruowanego (DPCM)')
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

blad = x - y
MSE = np.mean(blad ** 2)
print(f'Błąd średniokwadratowy (MSE): {MSE:.4f}')

plt.figure(figsize=(12, 4))
plt.plot(n, blad, label='Błąd rekonstrukcji', color='green')
plt.title('Wykres błędu rekonstrukcji DPCM')
plt.xlabel('Numer próbki')
plt.ylabel('Błąd amplitudy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

