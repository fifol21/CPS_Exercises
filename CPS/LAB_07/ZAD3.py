import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from test import fir_mono as fir_lpr

fs = 240000
pilot_freq = 19000
pilot_bw = 2000

w_pilot = (pilot_freq - pilot_bw/2) / (fs / 2), (pilot_freq + pilot_bw/2) / (fs / 2)
fir_pilot = signal.firwin(301, w_pilot, pass_zero=False, window='hamming')

T = 0.1
t = np.arange(0, T, 1/fs)
signal_fm = np.sin(2*np.pi*pilot_freq*t) + 0.5*np.sin(2 * np.pi * 38000 * t) + 0.2*np.random.randn(len(t))

pilot_out = signal.lfilter(fir_pilot, 1.0, signal_fm)

f_pilot, P_pilot =signal.welch(pilot_out, fs=fs, nperseg=2048)
plt.figure()
plt.semilogy(f_pilot, P_pilot)
plt.xlim(18000, 20000)
plt.title("Widmo gęstości mocy sygnału pilota")
plt.xlabel("Częstotliwość [Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()


fpl = f_pilot[np.argmax(P_pilot)]
f_lr = 2 * fpl
bw_lr = 4000
lr_low = (f_lr - bw_lr/2) / (fs / 2)
lr_high = (f_lr + bw_lr/2) / (fs / 2)
fir_lr = signal.firwin(301, [lr_low, lr_high], pass_zero=False, window='hamming')


lr_out = signal.lfilter(fir_lr, 1.0, signal_fm)
carrier = np.cos(2*np.pi*f_lr*t)
lr_baseband = lr_out * carrier


plt.figure()
for sig, label in zip([signal_fm, lr_out, lr_baseband], ['Wejściowy', 'Po filtrze BP', 'Po mnożeniu']):
    f, P = signal.welch(sig, fs=fs, nperseg=2048)
    plt.semilogy(f, P, label=label)
plt.title("Widma kolejnych etapów odzyskiwania sygnału stereo")
plt.xlabel("Częstotliwość [Hz]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()



lr_filtered_direct = signal.lfilter(fir_lr, 1.0, signal_fm)


carrier_cos = np.cos(2 * np.pi * f_lr * t)
lr_shifted = lr_filtered_direct * carrier_cos


plt.figure()
signals = [signal_fm, lr_filtered_direct, lr_shifted]
labels = ['Przed filtracją', 'Po filtracji', 'Po przesunięciu']
for sig, label in zip(signals, labels):
    f, P = signal.welch(sig, fs=fs, nperseg=2048)
    plt.semilogy(f, P, label=label)
plt.title("Widma sygnału w różnych etapach (punkt 3)")
plt.xlabel("Częstotliwość [Hz]")
plt.xlim(0, 80000)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Wyjaśnienie „ducha”:
# W widmie po przesunięciu można zaobserwować tzw. „ducha” w okolicach 4*fpl (czyli ~76kHz).
# Jest to efekt uboczny mnożenia rzeczywistego sygnału z kosinusem.
# Wynik takiej operacji zawiera dwa przesunięcia: do przodu i do tyłu w dziedzinie częstotliwości,
# co skutkuje dodatkowym obrazem sygnału na +2*f nośnej (czyli 2 * 2fpl).


#decreasing sampling frequency to 30kHz #4
from scipy.signal import decimate
lp_anty = signal.firwin(101, 15000 / (fs/2), window='hamming')
lr_filtered = signal.lfilter(lp_anty, 1.0, lr_baseband)

#decimation
y_lr_dec = decimate(lr_filtered, int(fs/30000), ftype='fir')

ym = signal.lfilter(fir_lpr, 1.0, signal_fm)
ys = lr_filtered

# compensating delay (rounded)
delay_mono = len(fir_lpr) // 2
delay_stereo = len(fir_lr) // 2 + len(lp_anty) // 2

ym_sync = ym[delay_stereo:len(ys)+delay_stereo]
ys_sync = ys[delay_mono:len(ym_sync)+delay_mono]

yl = 0.5 * (ym_sync + ys_sync)
yr = 0.5 * (ym_sync - ys_sync)

# base stereo chart
plt.figure(figsize=(10,4))
plt.plot(yl[:1000], label='Lewy')
plt.plot(yr[:1000], label='Prawy')
plt.title("Kanały stereo po rekonstrukcji")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()