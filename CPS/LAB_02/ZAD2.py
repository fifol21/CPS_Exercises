import numpy as np
from ZAD1 import DCT_matrix

N = 20
A = DCT_matrix(N)

############################################

S=A.T # IDCT
I = np.eye(N)
result_2 = np.allclose(S @ A , I , atol=1e-10)
print(f"A*S=I?  - {result_2}")

############################################
x=np.random.randn(N)
X= A @ x

x_s = S @ X

result_3 = np.allclose(x , x_s, atol=1e-12)
print(f"x*x_s - poprawna rekonstrukcja? - {result_3}")





