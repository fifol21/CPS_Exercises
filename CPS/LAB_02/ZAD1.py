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

N = 20;
A = DCT_matrix(N)
result = check_orto(A)
print(result)


