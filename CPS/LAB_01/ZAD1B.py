import numpy as np
import matplotlib.pyplot as plt

A=230
T=1
f=50

fs1= 10000
fs2=51
fs3=50
fs4=49


t1=np.arange(0, T, 1/fs1)
t2=np.arange(0,T, 1/fs2)
t3=np.arange(0,T, 1/fs3)
t4=np.arange(0,T, 1/fs4)

y1= A * np.sin(2*np.pi * f * t1)
y2= A * np.sin(2*np.pi * f * t2)
y3= A * np.sin(2*np.pi * f * t3)
y4= A * np.sin(2*np.pi * f * t4)

plt.title("ZADANIE_1_B")
plt.plot(t1,y1,"b-",label="10kHZ")
plt.plot(t2,y2,"g-o", label="51HZ") # fs - fn = 1hz
plt.plot(t3,y3,"r-o", label="50HZ")
plt.plot(t4,y4,"k-o", label="49HZ")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(V)")
plt.legend()
plt.grid(True)
plt.show()

# dla 49 hz powstanie zjawisko aliasingu