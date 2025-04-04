import numpy as np
import matplotlib.pyplot as plt

zeros = np.array([1j*5, -1j*5, 1j*15, -1j*15])
poles = np.array([-0.5 + 1j*9.5, -0.5 - 1j*9.5, -1 + 1j*10, -1 - 1j*10, -0.5 + 1j*10.5, -0.5 - 1j*10.5])

num = np.poly(zeros)
den = np.poly(poles)

w = np.linspace(0.1, 30, 1000)
jw = 1j * w
H_jw = np.polyval(num, jw) / np.polyval(den, jw)

plt.figure(figsize=(6, 6))
plt.scatter(zeros.real, zeros.imag, marker='o', color='blue', label='Zera')
plt.scatter(poles.real, poles.imag, marker='*', color='red', label='Bieguny')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.title('zeros and poles')
plt.grid()
plt.show()


plt.plot(w, np.abs(H_jw), label='|H(jw)|')
plt.xlabel('f')
plt.ylabel('A')
plt.title('a-cz')
plt.grid()
plt.legend()
plt.show()


dB_H_jw = 20 * np.log10(np.abs(H_jw))
plt.figure(figsize=(10, 6))
plt.plot(w, dB_H_jw, label='20log10|H(jw)|')
plt.xlabel('f ')
plt.ylabel('A')
plt.title('a-cz')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(w, np.angle(H_jw), label='Faza H(jw)')
plt.xlabel('f')
plt.ylabel('Faza')
plt.title('f-cz')
plt.grid()
plt.legend()
plt.show()
