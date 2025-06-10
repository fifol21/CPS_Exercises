import numpy as np

def entropy(x):
    symbols, counts = np.unique(x, return_counts=True) # znajdujemy symbole i liczymy ile razy wystepuja
    p = counts / counts.sum() # prawdopodobienstwo wystapienia kazdego symbolu
    H = -np.sum(p * np.log2(p)) # wzor na entoropie
    return H, symbols, p

x1 = [0, 1, 2, 3, 3, 2, 1, 0]
x2 = [0, 7, 0, 2, 0, 2, 0, 7, 4, 2]
x3 = [0, 0, 0, 0, 0, 0, 0, 15]

for i, x in enumerate([x1, x2, x3], start=1):
    H, symbols, p = entropy(x)
    print(f"x{i}:")
    print(f"  Unikalne symbole: {symbols}")
    print(f"  Prawdopodobieństwa: {p}")
    print(f"  Entropia H(x) = {H:.4f} bitów na symbol\n")
