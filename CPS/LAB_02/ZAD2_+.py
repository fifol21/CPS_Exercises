import random

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
print(f"ortonormality? {result_1}")
###
result_2 = np.allclose(A @ S ,I, atol=1e-10)
print(f"A*S=I? {result_2}")
###
X= A @ x

x_s = S @ X

result_3 = np.allclose(x , x_s, atol=1e-10)
print(f"x*x_s - {result_3}")
