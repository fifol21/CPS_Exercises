import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

#MDCT utility functions
def mdct_window(N):  # sin okno do wyhladzenia przejsc miedzy blokami
    n = np.arange(N)
    return np.sin(np.pi * (n + 0.5) / N)

def mdct_matrix(N):         # macierz MDCT o rozmiarze N ( pozwoli nam przejsc z t do f )
    k = np.arange(N // 2).reshape(-1,1)
    n = np.arange(N).reshape(1,-1)
    return np.sqrt(4 / N) * np.cos(2 * np.pi / N * (k + 0.5) * (n + 0.5 + N /4))

def frame_signal(signal, N): # tnie sygnał na nachodzące na siebie ramki
    hop = N //2
    pad = N- hop
    signal = np.pad(signal, (pad, pad), mode='constant')
    num_frames = (len(signal) - N) // hop + 1
    frames = np.stack([signal[i * hop:i * hop + N] for i in range(num_frames)])
    return frames, pad

def reconstruct_signal(frames, N, pad, original_length):   # sklada ramki spowrotem
    hop = N // 2
    signal_length = hop * (len(frames) +1)
    signal = np.zeros(signal_length)
    for i,frame in enumerate(frames):
        signal[i * hop:i * hop +N] += frame
    return signal[pad:pad + original_length]

def mdct_analysis(frames, A):
    return frames @ A.T

def mdct_synthesis(coeffs, A):
    return coeffs @ A

def quantize(x, Q):
    return np.round(x * Q)  # im wieksze Q tym mniejsza strata informacji ( opisuje nam czulosc)

def dequantize(y, Q):
    return y / Q


def process_mdct(signal, N, Q=None):
    A = mdct_matrix(N)
    window = mdct_window(N)     # generujemy macierz i okno MDCT

    frames, pad = frame_signal(signal, N)  # tniemy te ramki ktore sie na siebie nakladaja
    windowed = frames * window  # tutaj wykonujemy opcje nakladania sie okna - slim
    coeffs = mdct_analysis(windowed, A)  # obliczmy wspolczynnim MDCT
    if Q is not None:
        coeffs_q = quantize(coeffs, Q)
        coeffs = dequantize(coeffs_q, Q)
    reconstructed_frames = mdct_synthesis(coeffs, A)  # odwraca kwantyzacje MDCT
    windowed_back = reconstructed_frames * window
    reconstructed_signal = reconstruct_signal(windowed_back, N, pad, len(signal))  # sklada ramki spowrotem w pelny sygal
    return reconstructed_signal


def dynamic_bit_allocation(coeffs, max_bits = 16, min_bits = 2, window_ms = 100, fs=44100):
    frame_per_window = int(window_ms * fs / 1000 // (coeffs.shape[1]))
    frame_per_window = max(1, frame_per_window)

    bit_alloc_map = np.zeros_like(coeffs, dtype=int)
    for i in range(0, len(coeffs), frame_per_window):
        block = coeffs[i : i + frame_per_window]
        power = np.mean(np.abs(block), axis = 0)
        scaled = (power - power.min()) / (power.max() - power.min()+ 1e-9)
        bits = (scaled * (max_bits - min_bits) + min_bits).astype(int)
        bit_alloc_map[i:i+frame_per_window] = bits
    return bit_alloc_map

def quantize_dynamic(coeffs, bit_alloc):
    quantized = np.zeros_like(coeffs)
    Q_used = np.zeros_like(coeffs)
    for i in range(coeffs.shape[0]):
        bits = bit_alloc[i]
        Q = 2 ** bits
        Q_used[i] = Q
        quantized[i] = quantize(coeffs[i], Q)
    return quantized, Q_used

def dequantize_dynamic(quantized, Q_used):
    return quantized / Q_used

#reading audio file
fs, audio = wav.read('DontWorryBeHappy.wav')
if audio.ndim > 1:
    audio = np.mean(audio, axis = 1)
audio = audio / np.max(np.abs(audio))   #normalization

#test with quantization and without
for N in [32, 128]:
    #without quantization
    reconstructed_signal = process_mdct(audio, N)
    error = np.max(np.abs(reconstructed_signal - audio))
    print(f"MDCT without quantization, N={N}, max error: {error:.2e}")
    sf.write(f"reconstructed_N{N}_no_quant.wav", reconstructed_signal, fs)

    #with quantization
    Q = 100
    reconstructed_signal_q = process_mdct(audio, N, Q)
    error_q = np.max(np.abs(reconstructed_signal_q - audio))
    print(f"MDCT with quantization, N={N}, Q={Q}, max error: {error_q:.4f}")
    sf.write(f"reconstructed_N{N}_quant_Q{Q}.wav", reconstructed_signal_q, fs)

#dynamic fit allocation for N = 128
N = 128
A = mdct_matrix(N)
window = mdct_window(N)
frames, pad = frame_signal(audio, N)
windowed = frames * window
coeffs = mdct_analysis(windowed, A)

bit_alloc = dynamic_bit_allocation(coeffs, max_bits=6, min_bits=2, window_ms=100, fs=fs) # dzieli nasze bity na segmenty czasowe i analizuje energie w pasmach, wieksze zanczenie -> wiecej bitow
quantized, Qmap = quantize_dynamic(coeffs, bit_alloc) # dla kazdego wspolczynnika przydziela Q na podstawie liczby bitow
dequantized = dequantize_dynamic(quantized, Qmap)

reconstructed_frames = mdct_synthesis(dequantized, A)
windowed_back = reconstructed_frames * window
reconstructed_signal_dynamic = reconstruct_signal(windowed_back, N, pad, len(audio))
error_dynamic = np.max(np.abs(reconstructed_signal_dynamic - audio))
print(f"MDCT with dynamic bit allocation, N={N}, max error: {error_dynamic:.4f}")
sf.write("reconstructed_N128_dynamic.wav", reconstructed_signal_dynamic, fs)


