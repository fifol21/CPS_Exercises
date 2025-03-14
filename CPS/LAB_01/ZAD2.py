import numpy as np
import matplotlib.pyplot as plt

T = 0.1
fs = 10000
fs3 = 200
f= 50
t_analog = np.arange(0, T, 1/fs)
y_analog = np.sin(2*np.pi*f*t_analog) # oryginalna funkcja

t_sample = np.arange(0, T, 1/fs3)
y_sample = np.sin(2*np.pi*f*t_sample) # sprobkowany - nie wystapi aliasing bo spelnione jest tw nywuista

def sinc_interp(t_sample, y_sample, t_interp):
    T= 1/fs3  # okres "fs3"
    y_interp = np.zeros_like(t_interp)
    print(T)

    for i in range(len(y_sample)):
        y_interp += y_sample[i] * np.sinc((t_interp-t_sample[i])/T)

    return y_interp

t_interp = np.arange(0, T, 1/fs/5)
y_interp = sinc_interp(t_sample, y_sample, t_interp) # zrekonstruowany sygnal

y_analog_interp =np.sin(2*np.pi*f*t_interp)

error = y_analog_interp - y_interp

plt.figure(figsize=(10, 5))

plt.plot(t_interp, y_analog_interp, "b-", label="oryginalny")
plt.plot(t_interp, y_interp, "g-", label="rekonstrukcja")
plt.scatter(t_sample, y_sample, c="g", label="probki", zorder=10)
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_interp, error, "b-", label="błąd")
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Czas (s)")
plt.ylabel("Error")
plt.grid(True)
plt.show()


