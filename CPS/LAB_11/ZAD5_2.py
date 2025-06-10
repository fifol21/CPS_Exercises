import numpy as np
from math import log2
import heapq
from collections import Counter

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
        right = heapq.heappop(heap)
        merged = Node(prob=left.prob + right.prob, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]

# Tworzenie tablicy kodowej
def build_codebook(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codebook(node.left, prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook

# Dekodowanie ciągu binarnego
def decode(encoded_str, reverse_codebook):
    result = []
    current = ""
    for bit in encoded_str:
        current += bit
        if current in reverse_codebook:
            result.append(reverse_codebook[current])
            current = ""
    return result

# Wczytywanie pliku
with open("lab11.txt", "r") as f:
    data = [int(line.strip()) for line in f if line.strip().isdigit()]

print("Dane z pliku:", data)

# Obliczanie prawdopodobieństw
counts = Counter(data)
total = sum(counts.values())
probabilities = {k: v / total for k, v in counts.items()}

# Budowanie drzewa i kodera
root = build_huffman_tree(probabilities)
codebook = build_codebook(root)

# Kodowanie
encoded_bits = ''.join([codebook[sym] for sym in data])
print("\nZakodowana sekwencja:", encoded_bits)
print("Liczba bitów:", len(encoded_bits))

# Entropia
entropy = -sum(p * log2(p) for p in probabilities.values())
min_bits = entropy * len(data)
print(f"\nEntropia: {entropy:.4f} bity/symbol")
print(f"Minimalna liczba bitów: {min_bits:.2f}")

# Dekodowanie
reverse_codebook = {v: k for k, v in codebook.items()}
decoded = decode(encoded_bits, reverse_codebook)

print("\nZdekodowana sekwencja:", decoded)
print("Poprawność dekodowania:", decoded == data)