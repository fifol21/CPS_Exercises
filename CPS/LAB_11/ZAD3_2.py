import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

#  Prototyp filtru PQMF
def prototyp(L):
    a0=1; a1=-0.99998229; a2=0.99692250
    a4=1/np.sqrt(2); a6=np.sqrt(1-a2**2); a7=-np.sqrt(1-a1**2)
    A=-( a0/2+a1+a2+0+a4+a6+a7 )
    a3=A/2-np.sqrt(0.5-A**2/4)
    a5=-np.sqrt(1-a3**2)
    n = np.arange(L)
    p = a0*np.ones(L)
    for k, a in enumerate([a1, a2, a3, a4, a5, a6, a7], start=1):
        p += 2*a*np.cos(2*np.pi*k*n/L)
    p /= L
    p[0] = 0
    return p

# --- Analiza podpasmowa ---
def analysis_filter_bank(x, M):
    Nx = len(x)
    L = 16 * M
    MM = 2 * M
    Lp = L // MM

    p = prototyp(L)
    p = np.sqrt(M) * p

    for n in range(1, Lp, 2):
        start = n * MM
        end = start + MM
        p[start:end] = -p[start:end]

    m = np.arange(MM)
    A = np.zeros((M, MM))
    for k in range(M):
        A[k, :] = 2 * np.cos(np.pi / M * (k + 0.5) * (m - M / 2))

    K = Nx // M
    sb = np.zeros((M, K))

    bx = np.zeros(L)
    for k in range(K):
        bx = np.concatenate((x[k * M : (k + 1) * M][::-1], bx[:-M]))
        pbx = p * bx
        a = pbx.reshape((Lp, MM))
        u = np.sum(a, axis=0)
        sb[:, k] = A @ u

    return sb, A, p, Lp, M, K

# --- Dynamiczne przypisanie bitów ---
def dynamic_bit_allocation(sb, min_bits=3, max_bits=8):
    M, K = sb.shape
    q_bits = np.zeros((M, K), dtype=int)
    for k in range(K):
        energy = sb[:, k]**2
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-12)
        q_bits[:, k] = (energy_norm * (max_bits - min_bits) + min_bits).astype(int)
    return q_bits

# --- Kwantyzacja i dekwantyzacja ---
def quantize_dequantize(sb, q_bits):
    M, K = sb.shape
    sbq = np.zeros_like(sb)
    total_bits = 0
    for k in range(K):
        for m in range(M):
            q = 2 ** q_bits[m, k]
            max_val = np.max(sb[m, :])
            min_val = np.min(sb[m, :])
            if max_val == min_val:
                sbq[m, k] = sb[m, k]
                continue
            scaled = (sb[m, k] - min_val) / (max_val - min_val)
            quantized = np.round(scaled * (q - 1))
            dequantized = quantized / (q - 1) * (max_val - min_val) + min_val
            sbq[m, k] = dequantized
            total_bits += q_bits[m, k]

    bps = total_bits / (M * K * M)
    return sbq, bps

# === Główna część programu ===

fs, x = wavfile.read('DontWorryBeHappy.wav')
if x.ndim > 1:
    x = x[:,0]  # mono

start_sample = int(1*fs)
duration = int(0.5*fs)
x_frag = x[start_sample:start_sample+duration]
Nx = len(x_frag)

M = 32
sb, B, ap, Lp, M, K = analysis_filter_bank(x_frag, M)
q_bits = dynamic_bit_allocation(sb, min_bits=3, max_bits=8)
sbq, bps = quantize_dequantize(sb, q_bits)

# Spektrogram oryginalnego fragmentu
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
f, t, Sxx = spectrogram(x_frag, fs=fs, nperseg=256, noverlap=128)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spektrogram oryginału')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f dB')

# Spektrogram energii podpasm oryginalnych (sb^2)
plt.subplot(1,3,2)
plt.imshow(10*np.log10(sb**2 + 1e-12), aspect='auto', origin='lower',
           extent=[0, sb.shape[1]*M/fs, 0, M])
plt.colorbar(label='Energia [dB]')
plt.title('Energia podpasm oryginalnych')
plt.xlabel('Czas [s]')
plt.ylabel('Numer podpasma')

# Spektrogram energii podpasm kwantyzowanych (sbq^2)
plt.subplot(1,3,3)
plt.imshow(10*np.log10(sbq**2 + 1e-12), aspect='auto', origin='lower',
           extent=[0, sbq.shape[1]*M/fs, 0, M])
plt.colorbar(label='Energia [dB]')
plt.title('Energia podpasm kwantyzowanych')
plt.xlabel('Czas [s]')
plt.ylabel('Numer podpasma')

plt.tight_layout()
plt.show()

print(f"Średnia liczba bitów na próbkę (bps): {bps:.2f}")
