import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class SimpleADPCM:
    def __init__(self, bitrate=32):
        if bitrate == 16:
            self.levels = 4
        elif bitrate == 32:
            self.levels = 16
        else:
            raise ValueError("Obsługiwane: 16 lub 32 kbps")

        self.step = 1000.0    # krok kwantyzacji
        self.pred = 0.0       # predyktor

    def quantize(self, diff):
        level = int(round((diff / self.step) + (self.levels - 1) / 2))
        return max(0, min(self.levels - 1, level))

    def dequantize(self, level):
        return (level - (self.levels - 1) / 2) * self.step

    def encode(self, samples):
        encoded = []
        for s in samples:
            diff = float(s) - self.pred
            q = self.quantize(diff)
            dq = self.dequantize(q)

            self.pred += dq
            self.pred = np.clip(self.pred, -32768, 32767)

            # adaptacja kroku
            self.step = 0.9 * self.step + 0.1 * abs(dq)
            self.step = max(self.step, 1.0)

            encoded.append(q)
        return np.array(encoded, dtype=np.uint8)

    def decode(self, encoded):
        decoded = []
        for q in encoded:
            dq = self.dequantize(q)
            s = self.pred + dq
            s = np.clip(s, -32768, 32767)
            self.pred = s

            # adaptacja kroku
            self.step = 0.9 * self.step + 0.1 * abs(dq)
            self.step = max(self.step, 1.0)

            decoded.append(int(s))
        return np.array(decoded, dtype=np.int16)

if __name__ == "__main__":
    rate, data = wavfile.read("DontWorryBeHappy.wav")
    if data.ndim > 1:
        data = data[:, 0]  # tylko pierwszy kanał

    adpcm = SimpleADPCM(bitrate=32)
    encoded = adpcm.encode(data)

    decoder = SimpleADPCM(bitrate=32)
    decoded = decoder.decode(encoded)



    t = np.arange(len(data)) / rate
    plt.figure(figsize=(12, 6))

    plt.subplot(2,1,1)
    plt.plot(t, data)
    plt.title("Oryginalny sygnał")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(t, decoded, color='orange')
    plt.title("Sygnał po kodowaniu i dekodowaniu ADPCM")
    plt.grid()

    plt.tight_layout()
    plt.show()
