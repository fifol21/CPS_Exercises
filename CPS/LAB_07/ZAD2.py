import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz


Fs = 192000
nyq = Fs / 2

cutoff_mono = 15000
stopband_start = 19000
attenuation_db = 40


numtaps_mono = 501

fir_mono = firwin(numtaps=numtaps_mono,cutoff=cutoff_mono, window='hamming',fs=Fs)


pilot_freq = 19000
bandwidth = 500
lowcut = pilot_freq - bandwidth / 2
highcut = pilot_freq + bandwidth / 2

numtaps_pilot = 1001
fir_pilot = firwin( numtaps=numtaps_pilot,cutoff=[lowcut, highcut],window='blackmanharris',pass_zero=False, fs=Fs) # blackman chyba tez bedzie ok, ale ten daje mniej zafalowan na zboczach


def plot_filter_response(fir, label, Fs):
    w, h = freqz(fir, worN=8000, fs=Fs)
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=label)
    plt.xlabel(" [Hz]")
    plt.ylabel(" [dB]")
    plt.grid(True)
    plt.ylim(-100, 5)

plt.figure(figsize=(12, 6))
plot_filter_response(fir_mono, "Mono ", Fs)
plot_filter_response(fir_pilot, "Pilot 19kHz ", Fs)
plt.axvline(19000, color='red', linestyle='--', label="Pilot 19kHz")
plt.axvline(15000, color='green', linestyle='--', label="Mono 15kHz")
plt.legend()
plt.title("FIR")
plt.show()
