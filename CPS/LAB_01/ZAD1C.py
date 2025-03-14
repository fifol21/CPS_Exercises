import numpy as np
import matplotlib.pyplot as plt

fs = 100
T = 1
t=np.arange(0,T,1/fs)

for i in range(61):
    f = i * 5
    y = np.sin(2*np.pi * f * t) #cos
    plt.plot(t,y,label="f")
    plt.title(f"Obieg {i}, {f} HZ")
    plt.show()
    plt.grid()

    print(f"Obieg {i + 1}, Częstotliwość = {f} Hz")

for f in np.array([5,105,205]):
    plt.plot(t,np.sin(2*np.pi * f * t),label=f"{f}")
    plt.title(f"5,105,205")
    plt.grid()
plt.show()

for f in np.array([95,105]):
    plt.plot(t,np.sin(2*np.pi * f * t),label=f"{f}")
    plt.title(f"95,105")
    plt.grid()
plt.show()





