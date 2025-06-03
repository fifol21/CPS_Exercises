import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as signal

fs_ekg = 360


mat = sio.loadmat("ECG100.mat")
x = mat['val'][0]


f_ekg, Pxx_ekg = signal.welch(x, fs=fs_ekg, nperseg=1024)
plt.figure()
plt.semilogy(f_ekg, Pxx_ekg)
plt.title("Widmo sygnału EKG")
plt.xlabel("Częstotliwość [Hz]") # patrzymy sobie jakie czestotliwosci domnuja w sygnale, jakue sa zaklocenia
plt.grid()
plt.tight_layout()
plt.show()


f_cut = 40
fir_ekg = signal.firwin(101, f_cut / (fs_ekg / 2), window='blackman')


y = signal.lfilter(fir_ekg, 1.0, x)
delay = len(fir_ekg) // 2
x_sync = x[delay:len(y)] # poprostsu ucinamy czesc tych probek tak zeby byly zsynchronizowane w czasiem, inaczej linie na wykresie by mogly byc przesuniete
y_sync = y[delay:]

plt.figure()
plt.plot(x_sync[:1000], label='Oryginalny')
plt.plot(y_sync[:1000], label='Po filtracji')
plt.title("Odszumianie EKG - filtr FIR")
plt.xlabel("Próbka")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


np.random.seed(0)
x_noisy = x + 0.5 * np.random.randn(len(x))
y_noisy = signal.lfilter(fir_ekg, 1.0, x_noisy)
yn_sync = y_noisy[delay:]
xn_sync = x[delay:len(yn_sync)+delay]

plt.figure()
plt.plot(xn_sync[:1000], label='Czysty EKG')
plt.plot(yn_sync[:1000], label='Po odszumianiu')
plt.title("Odszumianie zaszumionego sygnału EKG")
plt.xlabel("Próbka")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


firs = {
    'wąski (20Hz)': signal.firwin(101, 20 / (fs_ekg / 2), window='hamming'),
    'szeroki (70Hz)': signal.firwin(101, 70 / (fs_ekg / 2), window='hamming'),
    'długi (40Hz, M=201)': signal.firwin(201, 40 / (fs_ekg / 2), window='hamming') # blackman moze byc, ale kosztem tlumienia poszerzy nam sie listek glowny
} # a glownie chodzi o to zeby listek byl waski zeby dobtze odczytac wartosci EKG

noise_levels = [0.2, 0.5, 1.0]

for name, taps in firs.items():
    for noise_amp in noise_levels:
        x_noise = x + noise_amp * np.random.randn(len(x))
        y_noise = signal.lfilter(taps, 1.0, x_noise)
        delay = len(taps) // 2
        y_sync = y_noise[delay:]
        x_sync = x[delay:delay + len(y_sync)]
        plt.figure()
        plt.plot(x_sync[:1000], label='Czysty')
        plt.plot(y_sync[:1000], label=f'Odfiltrowany ({name}, szum={noise_amp})')
        plt.title(f"Filtr: {name}, szum={noise_amp}")
        plt.xlabel("Próbka")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
