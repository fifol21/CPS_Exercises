import numpy as np
import matplotlib.pyplot as plt

A = 230
f = 50
T = 0.1

f1= 10000
f2 = 500
f3 = 200

t1=np.arange(0,T,1/f1)
t2=np.arange(0,T,1/f2) # test gita
t3=np.arange(0,T,1/f3)

x1= A*np.sin(2*np.pi*f*t1)
x2= A*np.sin(2*np.pi*f*t2)
x3= A*np.sin(2*np.pi*f*t3)

plt.plot(t1,x1,"b-",label="10kHZ")
plt.plot(t2,x2,"r-o", label="500HZ")
plt.plot(t3,x3,"k-o", label="200HZ")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude(V)")
plt.legend()
plt.grid(True)
plt.show()

# Wniosek- im wieksza czestotwlisc probkowanie tym lepsze odwzorowanie, im mniejsza tym slabsze i moze powstac efekt aliasingu
# musi byc spelnione kryterium nyquista
