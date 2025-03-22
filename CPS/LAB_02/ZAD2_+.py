
import numpy as np
# data
N = 20
A = np.random.randn(N,N)
S = np.linalg.inv(A)
I = np.eye(N)
x = np.random.randn(N)

#analisys

rows,cols = A.shape
norm = np.linalg.norm(A, axis=1)
result_1 = np.allclose(norm, 1, atol=1e-10)
print(f"normality =1 ? {result_1}") # probably false cuz matrix is random


result_2 = np.allclose(A @ S ,I, atol=1e-10)
print(f"A*S=I? {result_2}")

X= A @ x
x_s = S @ X

result_3 = np.allclose(x , x_s, atol=1e-10)
print(f"x*x_s - perfect reconstuction? - {result_3}")

def DCT_matrix_corrupted(N):
    A=np.zeros((N,N))
    for k in range(N):
        s_k = np.sqrt(1/N) if k==0 else np.sqrt(2/N)
        for n in range(N):
            A[k,n] = s_k * np.cos((np.pi*k+0.25/N )*(n+0.5)) #added corruption
    return A

A_2 = DCT_matrix_corrupted(N)
############################################################

S_2=A_2.T # IDCT
I_2 = np.eye(N)
result_4 = np.allclose(S_2 @ A_2 , I_2 , atol=1e-10)
print(f"A*S=I? (corrupted)- {result_4}")
############################################################

x_2= np.random.randn(N)
X_2= A_2 @ x_2

x_s_2 = S_2 @ X_2

result_5 = np.allclose(x_2 , x_s_2, atol=1e-10)
print(f"x_2*x_s_2- perfect reconstruction? (corrupted) - {result_5}")