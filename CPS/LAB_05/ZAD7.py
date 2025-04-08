import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametry oryginalnego filtru (projektowane wartości)
R1_orig = R2_orig = 3900  # 3.9 kΩ
C1_orig = C2_orig = 680e-12  # 680 pF
R3_orig = 2200  # 2.2 kΩ
C3_orig = 470e-12  # 470 pF

# Parametry dopasowane do typoszeregu E24
# Zakładamy dostępność tylko E24 (więc zaokrąglamy do najbliższej wartości z E24)
R1_mod = R2_mod = 3900  # bez zmian, 3.9k to E24
C1_mod = C2_mod = 680e-12  # bez zmian, 680pF to E24
R3_mod = 2200  # bez zmian
C3_mod = 470e-12  # bez zmian


# Tworzymy funkcję do obliczania transmitancji Sallen-Key + RC (łącznie 3 rzędu)
def total_transfer_function(R1, R2, C1, C2, R3, C3):
    # Sallen-Key 2nd order low-pass
    num_sk = [1]
    den_sk = [R1 * R2 * C1 * C2, (R1 + R2) * C2, 1]

    # RC 1st order low-pass
    num_rc = [1]
    den_rc = [R3 * C3, 1]

    # Łączna transmitancja: H_total = H1 * H2
    num_total = np.polymul(num_sk, num_rc)
    den_total = np.polymul(den_sk, den_rc)

    return signal.TransferFunction(num_total, den_total)


# Transmitancje
H_orig = total_transfer_function(R1_orig, R2_orig, C1_orig, C2_orig, R3_orig, C3_orig)
H_mod = total_transfer_function(R1_mod, R2_mod, C1_mod, C2_mod, R3_mod, C3_mod)

# Częstotliwości do analizy (0 Hz do 500 kHz)
w, mag_orig, _ = signal.bode(H_orig, w=np.logspace(3, 6, 1000))
_, mag_mod, _ = signal.bode(H_mod, w=w)

# Wykres
plt.figure(figsize=(10, 6))
plt.semilogx(w / (2 * np.pi), mag_orig, label='Oryginalna charakterystyka')
plt.semilogx(w / (2 * np.pi), mag_mod, '--', label='Po dopasowaniu do E24')
plt.axhline(-3, color='red', linestyle=':', label='Granica 3 dB')
plt.title('Charakterystyki częstotliwościowe filtru')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Wzmocnienie [dB]')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()

# Sprawdź czy różnica mieści się w granicy 3 dB
delta = np.abs(np.array(mag_orig) - np.array(mag_mod))
np.all(delta <= 3)  # zwróci True, jeśli różnice są w dopuszczalnym zakresie

