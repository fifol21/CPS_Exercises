import numpy as np

def DCT_matrix(N):
    A=np.zeros((N,N))
    for k in range(N):
        s_k = np.sqrt(1/N) if k==0 else np.sqrt(2/N)
        for n in range(N):
            A[k,n] = s_k * np.cos((np.pi*k/N )*(n+0.5))
    return A

def check_orto(A):
    rows,cols = A.shape
    if rows!=cols:
        return False
    else:
        I = np.eye(rows)
        return np.allclose(A.T @ A , I , atol=1e-10)

N = 20
A = DCT_matrix(N)


############################################
S=A.T # IDCT
I = np.eye(N)
result_2 = np.allclose(S @ A , I , atol=1e-10)
print(f"A*S=I - {result_2}")


############################################
x=np.random.randn(N)
X= A @ x

x_s = S @ X

result_3 = np.allclose(x , x_s, atol=1e-10)
print(f"x*x_s - poprawna rekonstrukcja ? - {result_3}")





