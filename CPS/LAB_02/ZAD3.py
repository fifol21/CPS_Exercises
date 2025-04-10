import numpy as np
import matplotlib.pyplot as plt
from ZAD1 import DCT_matrix
N = 100
fs = 1000

f1, f2, f3 = 50, 100, 150 #100 - f2
f1_alt , f2_alt, f3_alt = 52.5, 107.5, 152.5
A1, A2 , A3 = 50 , 100, 150
t= np.arange(N)/fs

x= A1*np.cos(2*np.pi*f1*t)+A2*np.cos(2*np.pi*f2*t)+A3*np.cos(2*np.pi*f3*t)
x_alt = A1*np.cos(2*np.pi*f1_alt*t)+A2*np.cos(2*np.pi*f2_alt*t)+A3*np.cos(2*np.pi*f3_alt*t)

A = DCT_matrix(N)
S = A.T

for i in range(N):
    plt.figure()
    plt.plot(A[i,:],"bo", label=f"wiersz {i+1} A")
    plt.plot(S[:,i],"r--", label=f"kolumna{i+1} S")
    plt.legend()
    plt.pause(0.02)
    plt.close()
plt.show()

f = (np.arange(N) * fs) / (2 * N)
y = A @ x
y_alt = A @ x_alt
plt.subplot(2, 1, 1)
plt.stem(f,y,linefmt='b-',  basefmt="black")
plt.grid(True)
plt.xticks(np.arange(0, fs / 2, step=50))
plt.title("DCT_perfect index")
plt.ylabel("index")
plt.xlabel("f")

plt.subplot(2, 1, 2)
plt.stem(f,y_alt,linefmt='r-',  basefmt="black")
plt.grid(True)
plt.xticks(np.arange(0, fs / 2, step=50))
plt.title("DCT_moved index")
plt.ylabel("index")
plt.xlabel("f")
plt.tight_layout()

plt.show()


values = y[1:N]
tol = 1e-6
for i, val in enumerate(values):
    if abs(val) > tol:
        print(f"Value {val} -> {y[i]}")
        print(f"index{i} ->  {f[i]}")

x_r = S @ y
result = np.allclose(x_r, x, atol=1e-10)
print(f"x*x_r - {result}")

