from math import log10

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


N_values = [2,4,6,8]

f = np.linspace(1, 10 * 100, 1000)
w = 2 * np.pi * f
w_3dB = np.pi*100

## a-cz liniowo
for N in N_values:
    poles = []
    for k in range(1,N+1):
        p_k = w_3dB * np.exp(1j * ((np.pi / 2) + (0.5 * np.pi / N) + (k-1) * np.pi / N))
        poles.append(p_k)

    a = np.poly(poles) #bieguny
    b=[1.0] # zera - wartosc w tym filtrze rowna 0

    w, H = signal.freqs(b, a, w)  # analiza czestotilowosciowaq

    plt.plot(f, 20*np.log10(np.abs(H)))
plt.xlabel('f ')
plt.ylabel('A')
plt.title("A-cz")
plt.grid(True)
plt.show()


## a-czk log
for N in N_values:
    poles = []
    for k in range(1, N + 1):
        p_k = w_3dB * np.exp(1j * ((np.pi / 2) + (0.5 * np.pi / N) + (k - 1) * np.pi / N))
        poles.append(p_k)

    a = np.poly(poles)
    b = [1.0]
    w, H = signal.freqs(b, a, w)  # analiza czestotilowosciowaq

    plt.semilogx(f, 20*np.log10(np.abs(H)))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.title("A-cz (log0)")
plt.grid(True)
plt.show()

## fazowa
for N in N_values:
    poles = []
    for k in range(1, N + 1):
        p_k = w_3dB * np.exp(1j * ((np.pi / 2) + (0.5 * np.pi / N) + (k - 1) * np.pi / N))
        poles.append(p_k)

    a = np.poly(poles)
    b = [1.0]
    w, H = signal.freqs(b, a, w)  # analiza czestotilowosciowaq

    plt.plot(f, np.angle(H), label=f'Angle(H){N}')

plt.xlabel('Frequency [Hz]')
plt.ylabel('phase [rad]')
plt.legend()
plt.grid(True)
plt.show()

N = 4
poles = []
for k in range(1, N + 1):
    p_k = w_3dB * np.exp(1j * ((np.pi / 2) + (0.5 * np.pi / N) + (k - 1) * np.pi / N))
    poles.append(p_k)

a = np.poly(poles)
b = [1.0]


system = signal.TransferFunction(b, a)
t , impulse = signal.impulse(system)
plt.plot(t, impulse)
plt.xlabel('Time [s]')
plt.ylabel('Impulse')
plt.grid(True)
plt.show()

t,step = signal.step(system)
plt.plot(t, step)
plt.xlabel('Time [s]')
plt.ylabel('Step')
plt.grid(True)
plt.show()















