import numpy as np
from math import log2
import heapq

# Definicja węzła drzewa Huffmana
class Node:
    def __init__(self, symbol=None, prob=0.0, left=None, right=None):
        self.symbol = symbol
        self.prob = prob
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.prob < other.prob

# Budowanie drzewa Huffmana
def build_huffman_tree(probabilities):
    heap = [Node(symbol=sym, prob=prob) for sym, prob in probabilities.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap) # dwa lacza z nadrzednym
        merged = Node(prob=left.prob + right.prob, left=left, right=right) # nowy wezel -> prawdop= suma prawdop dzieci
        heapq.heappush(heap, merged)

    return heap[0]  # korzeń

# Tworzenie słownika kodowego
def build_codebook(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.symbol is not None:
        codebook[node.symbol] = prefix # jezeli istnieje -> przypisujemy mu sciezke bo to koncowy sybol lisc 
    else:
        build_codebook(node.left, prefix + "0", codebook) # jesli to nie lisc \(wezel)\  dopisujemy lewo lub prawo
        build_codebook(node.right, prefix + "1", codebook)

    return codebook

# Wizualizacja drzewa
def print_tree(node, prefix=""):
    if node.symbol is not None:
        print(f"{prefix}Symbol: {node.symbol}, Prawdopodobieństwo: {node.prob}")
    else:
        print(f"{prefix}Węzeł: Prawdopodobieństwo: {node.prob}")
        print_tree(node.left, prefix + "  0-> ")
        print_tree(node.right, prefix + "  1-> ")

# 1. Dane
probabilities = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}

# 2. Budowanie drzewa
root = build_huffman_tree(probabilities)

# 3. Drukowanie drzewa
print("Drzewo Huffmana:")
print_tree(root)

# 4. Generowanie tablicy kodowej
codebook = build_codebook(root)
print("\nTablica kodowa (symbol -> kod):")
for symbol, code in sorted(codebook.items()):
    print(f"Symbol {symbol}: {code}")

# 5. Generowanie x4
np.random.seed(0)
x4 = np.random.randint(1, 5, size=10)
print("\nSygnał x4:", x4)

# 6. Kodowanie
encoded_bits = ''.join([codebook[symbol] for symbol in x4])
print("Zakodowana sekwencja:", encoded_bits)
print("Liczba bitów:", len(encoded_bits))

# 7. Entropia
entropy = -sum(p * log2(p) for p in probabilities.values())
min_bits = entropy * len(x4)
print(f"Entropia (bity na symbol): {entropy:.4f}")
print(f"Minimalna liczba bitów do zakodowania x4: {min_bits:.2f}")

# 8. Dekodowanie
reverse_codebook = {v: k for k, v in codebook.items()}

def decode(encoded_str, reverse_codebook):
    result = []
    current = ""
    for bit in encoded_str:
        current += bit
        if current in reverse_codebook:
            result.append(reverse_codebook[current])
            current = ""
    return result

decoded_x4 = decode(encoded_bits, reverse_codebook)
print("Zdekodowana sekwencja:", decoded_x4)
print("Oryginał == Dekodowane:", list(x4) == decoded_x4)


print("\nPrzekłamanie pierwszego bitu: ")
print(encoded_bits)
# Zmiana pierwszego bitu w zakodowanym ciągu
if encoded_bits[0] == '0':
    encoded_bits = '1' + encoded_bits[1:]
else:
    encoded_bits = '0' + encoded_bits[1:]
print(encoded_bits)

print("\ndekodowanie")
decoded_x4 = decode(encoded_bits, reverse_codebook)
print(decoded_x4)

print("jak widać pierwsze 2 symbole sostały dotknięte przez przekłamanie jednego bitu!!!")