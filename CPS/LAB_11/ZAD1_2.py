import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def adpcm_encode(x, bits=2):
    levels = 2 ** bits
    step = np.std(x) / 2 # krok kwantyzacji jako polowa odchylenia standardowego
    a = 0.9545
    y_prev = 0 # prevoius
    codes = []

    for sample in x:
        pred = a * y_prev
        diff = sample - pred
        index = int(np.clip(np.round(diff / step + (levels / 2)), 0, levels - 1))
        dq = (index - levels // 2) * step
        codes.append(index)
        y_prev = pred + dq

    return np.array(codes), step

def adpcm_decode(codes, step, bits=2):
    levels = 2 ** bits
    a = 0.9545
    y_prev = 0
    y = []

    for code in codes:
        dq = (code - levels // 2) * step
        y_curr = a * y_prev + dq
        y.append(y_curr)
        y_prev = y_curr

    return np.array(y)

# Wczytaj sygnał
fs, x = wavfile.read('DontWorryBeHappy.wav')
x = x.astype(np.float64)
if x.ndim > 1:
    x = x[:, 0]  # tylko pierwszy kanał

# ADPCM 2-bit
codes_2bit, step2 = adpcm_encode(x, bits=2)
y_2bit = adpcm_decode(codes_2bit, step2, bits=2)

# ADPCM 4-bit
codes_4bit, step4 = adpcm_encode(x, bits=4)
y_4bit = adpcm_decode(codes_4bit, step4, bits=4)

# MSE
mse_2bit = np.mean((x - y_2bit) ** 2)
mse_4bit = np.mean((x - y_4bit) ** 2)
print("MSE dla 2-bitowego ADPCM :", mse_2bit)
print("MSE dla 4-bitowego ADPCM :", mse_4bit)

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(x, label='Oryginalny', color='blue', alpha=0.5)
plt.plot(y_2bit, label='ADPCM 2-bit', color='red', linestyle='--')
plt.plot(y_4bit, label='ADPCM 4-bit', color='green', linestyle=':')
plt.legend()
plt.title('ADPCM - porównanie')
plt.xlabel('Próbka')
plt.ylabel('Amplituda')
plt.grid(True)
plt.tight_layout()
plt.show()
