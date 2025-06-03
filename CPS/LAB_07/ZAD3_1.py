import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io

# Parametry
fs = 240000
pilot_freq = 19000
pilot_bw = 2000
bw_lr = 4000

# Filtrowanie pilota (19 kHz)
w_pilot = ((pilot_freq - pilot_bw / 2) / (fs / 2), (pilot_freq + pilot_bw / 2) / (fs / 2))
fir_pilot = signal.firwin(301, w_pilot, pass_zero=False, window='hamming')

# Filtrowanie L-R (38 kHz)
fpl = 19000
f_lr = 2 * fpl
lr_low = (f_lr - bw_lr / 2) / (fs / 2)
lr_high = (f_lr + bw_lr / 2) / (fs / 2)
fir_lr = signal.firwin(301, [lr_low, lr_high], pass_zero=False, window='hamming')

# Filtr dolnoprzepustowy antyaliasingowy
fir_lpr = signal.firwin(301, 15000 / (fs / 2), window='hamming')
lp_anty = signal.firwin(101, 15000 / (fs / 2), window='hamming')

# ===== Sygnał syntetyczny z .mat =====
data = scipy.io.loadmat('stereo_samples_fs1000kHz_LR_IQ.mat')
I = data['I'].squeeze()
Q = data['Q'].squeeze()
iq = I + 1j * Q
fs_IQ = 1000000
fm_center = 250000
t_iq = np.arange(len(iq)) / fs_IQ

# Przesunięcie do 0 Hz i demodulacja
iq_shifted = iq * np.exp(-1j * 2 * np.pi * fm_center * t_iq)
diff_phase = np.angle(iq_shifted[1:] * np.conj(iq_shifted[:-1]))

fs_fm = fs_IQ // 5  # 200 kHz
fm_demod = signal.decimate(diff_phase, 5, ftype='fir')

# Ponowna interpolacja do 240 kHz dla reszty dekodera
factor_up = 6
fm_resampled = signal.resample_poly(fm_demod, factor_up, 5)
fs_resampled = fs_fm * factor_up  # = 240000 Hz
t = np.arange(len(fm_resampled)) / fs_resampled

# Pilot
pilot_out = signal.lfilter(fir_pilot, 1.0, fm_resampled)

# L-R
lr_out = signal.lfilter(fir_lr, 1.0, fm_resampled)
carrier = np.cos(2 * np.pi * f_lr * t)
lr_baseband = lr_out * carrier

# Antyaliasing i decymacja
lr_filtered = signal.lfilter(lp_anty, 1.0, lr_baseband)
y_lr_dec = signal.decimate(lr_filtered, int(fs_resampled / 30000), ftype='fir')

# L+R
ym = signal.lfilter(fir_lpr, 1.0, fm_resampled)
ys = lr_filtered

delay_mono = len(fir_lpr) // 2
delay_stereo = len(fir_lr) // 2 + len(lp_anty) // 2

ym_sync = ym[delay_stereo:len(ys) + delay_stereo]
ys_sync = ys[delay_mono:len(ym_sync) + delay_mono]

yl = 0.5 * (ym_sync + ys_sync)
yr = 0.5 * (ym_sync - ys_sync)

plt.figure(figsize=(10, 4))
plt.plot(yl[:1000], label='Lewy')
plt.plot(yr[:1000], label='Prawy')
plt.title("Kanały stereo (sygnał syntetyczny)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


fs_raw = 3200000
samples = np.fromfile('samples_100MHz_fs3200kHz.raw', dtype=np.int8).astype(np.float32)
samples -= np.mean(samples)
samples /= np.max(np.abs(samples))

analytic = signal.hilbert(samples)
diff_phase_real = np.angle(analytic[1:] * np.conj(analytic[:-1]))

fs_fm_real = fs_raw // 8
fm_demod_real = signal.decimate(diff_phase_real, 8, ftype='fir')

# Resampling do 240 kHz
fm_resampled_real = signal.resample_poly(fm_demod_real, 3, 1)
fs_resampled_real = fs_fm_real * 3  # = 240000 Hz
t_real = np.arange(len(fm_resampled_real)) / fs_resampled_real

# Dekodowanie stereo
pilot_out = signal.lfilter(fir_pilot, 1.0, fm_resampled_real)
lr_out = signal.lfilter(fir_lr, 1.0, fm_resampled_real)
carrier = np.cos(2 * np.pi * f_lr * t_real)
lr_baseband = lr_out * carrier

lr_filtered = signal.lfilter(lp_anty, 1.0, lr_baseband)
y_lr_dec = signal.decimate(lr_filtered, int(fs_resampled_real / 30000), ftype='fir')

ym = signal.lfilter(fir_lpr, 1.0, fm_resampled_real)
ys = lr_filtered

delay_mono = len(fir_lpr) // 2
delay_stereo = len(fir_lr) // 2 + len(lp_anty) // 2

ym_sync = ym[delay_stereo:len(ys) + delay_stereo]
ys_sync = ys[delay_mono:len(ym_sync) + delay_mono]

yl = 0.5 * (ym_sync + ys_sync)
yr = 0.5 * (ym_sync - ys_sync)

plt.figure(figsize=(10, 4))
plt.plot(yl[:1000], label='Lewy (real)')
plt.plot(yr[:1000], label='Prawy (real)')
plt.title("Kanały stereo (sygnał rzeczywisty)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


E_L = np.sum(yl ** 2)
E_R = np.sum(yr ** 2)

yl_nc = 0.5 * (ym[:len(ys)] + ys[:len(ym)])
yr_nc = 0.5 * (ym[:len(ys)] - ys[:len(ym)])
E_L_nc = np.sum(yl_nc ** 2)
E_R_nc = np.sum(yr_nc ** 2)

cross_talk_L = 10 * np.log10(E_R / E_L)
cross_talk_L_nc = 10 * np.log10(E_R_nc / E_L_nc)

print(f"Energia L: {E_L}, Energia R: {E_R}")
print(f"Energia L (bez korekcji): {E_L_nc}, Energia R (bez korekcji): {E_R_nc}")
print(f"Przesłuch z korekcją opóźnienia: {cross_talk_L} dB")
print(f"Przesłuch bez korekcji: {cross_talk_L_nc} dB")

print(f"Przesłuch z korekcją opóźnienia: {cross_talk_L} dB")
print(f"Przesłuch bez korekcji: {cross_talk_L_nc} dB")
