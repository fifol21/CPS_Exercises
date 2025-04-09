import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, freqs

# Funkcja pomocnicza: zaokrąglanie do szeregu E24 (1% tolerancji)
def round_to_e24(value_k):
    E24 = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])
    decades = 10 ** np.floor(np.log10(value_k))
    value_norm = value_k / decades
    closest = E24[np.argmin(np.abs(E24 - value_norm))]
    return closest * decades

# Oryginalne wartości elementów (kΩ i nF)
R1 = R2 = R3 = 18.13e3
R1B = 13.82e3
R2B = 3.8197e3
C = 1e-9

# Zaokrąglone do E24
R1r = R2r = R3r = round_to_e24(R1 / 1e3) * 1e3
R1Br = round_to_e24(R1B / 1e3) * 1e3
R2Br = round_to_e24(R2B / 1e3) * 1e3

# Funkcja przenoszenia – aproksymacja: 2x Sallen-Key + 1 RC
def build_butterworth_5th_order(R1, R1B, R2, R2B, R3, C):
    # Sallen-Key 1
    w01 = 1 / (R1 * C)
    Q1 = R1B / (2 * R1)
    # Sallen-Key 2
    w02 = 1 / (R2 * C)
    Q2 = R2B / (2 * R2)
    # Ostatni biegun RC
    w0r = 1 / (R3 * C)

    # Całkowita transmitancja – przybliżenie 5. rzędu Butterwortha
    num = [1]
    den = np.polymul(
        [1/(w01**2), 1/(w01*Q1), 1],
        np.polymul(
            [1/(w02**2), 1/(w02*Q2), 1],
            [1/w0r, 1]
        )
    )
    return TransferFunction(num, den)

# Oryginalny i zaokrąglony filtr
H_orig = build_butterworth_5th_order(R1, R1B, R2, R2B, R3, C)
H_rounded = build_butterworth_5th_order(R1r, R1Br, R2r, R2Br, R3r, C)

# Częstotliwości: logarytmicznie od 10 Hz do 10 MHz
w = np.logspace(1, 7, 1000)
w_rad = 2 * np.pi * w

# Odpowiedź częstotliwościowa
_, mag_orig, _ = H_orig.bode(w_rad)
_, mag_rnd, _ = H_rounded.bode(w_rad)

# Rysowanie
plt.figure(figsize=(10, 6))
plt.semilogx(w, mag_orig, label="Oryginalny")
plt.semilogx(w, mag_rnd, label="Zaokrąglony do E24", linestyle='--')
plt.axhline(-3, color='gray', linestyle=':', label="-3 dB")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.title("Charakterystyki częstotliwościowe filtra Butterwortha 5. rzędu")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()

# Sprawdzenie maksymalnego odchylenia
diff = np.abs(mag_orig - mag_rnd)
max_diff = np.max(diff)
print(f"Maksymalna różnica charakterystyki: {max_diff:.2f} dB")
print(" Mieści się w 3 dB" if max_diff <= 3 else "Przekracza 3 dB")
