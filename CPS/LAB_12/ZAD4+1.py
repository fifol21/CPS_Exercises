import numpy as np
from imageio.v2 import imread
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os
from PIL import Image


# DCT/IDCT 2D
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# Kwantyzacja i odwrotna
def quantize(block, q):
    return np.round(block / q)


def dequantize(block, q):
    return block * q


# PSNR
def psnr(original, reconstructed, b=8):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_i = 2 ** b - 1
    return 10 * np.log10((max_i ** 2) / mse)


# Dzielenie i składanie bloków
def block_split(img, size=8):
    h, w = img.shape
    return [img[i:i + size, j:j + size] for i in range(0, h, size) for j in range(0, w, size)]


def block_merge(blocks, shape, size=8):
    h, w = shape
    img = np.zeros(shape)
    idx = 0
    for i in range(0, h, size):
        for j in range(0, w, size):
            img[i:i + size, j:j + size] = blocks[idx]
            idx += 1
    return img


# Zigzag i odwrotnie
def zigzag(input):  # wyzerowanie wyższysz f, zapisanie niższych w jednej wektorze
    h, w = input.shape
    result = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < h and j < w:
                    result.append(input[i][j])
        else:
            for i in range(s + 1):
                j = s - i
                if j < h and i < w:
                    result.append(input[j][i])
    return result


def inverse_zigzag(input):  # wektor z zig zaka w postaci macierzy
    output = np.zeros((8, 8))
    order = np.array(zigzag(np.arange(64).reshape(8, 8)))
    for k, v in enumerate(input):
        idx = np.where(order == k)[0][0]
        i, j = divmod(idx, 8)
        output[i, j] = v
    return output


# RLE i odwrotność
def rle(arr):  # zapisuje w formacie : ile zer znajduje się przed daną liczbą , na koncu (0,0) czyli do końca same zera
    result = []
    zero_count = 0
    for val in arr:
        if val == 0:
            zero_count += 1
        else:
            result.append((zero_count, int(val)))
            zero_count = 0
    result.append((0, 0))  # EOB
    return result


def irle(pairs):  # odwrotność rle
    result = []
    for zeros, val in pairs:
        if (zeros, val) == (0, 0):
            break
        result.extend([0] * zeros)
        result.append(val)
    while len(result) < 63:
        result.append(0)
    return result


# Kodowanie JPEG
def jpeg_encode(img, q):  # łączy wszystkie funkcje przygotowane wcześniej
    blocks = block_split(img)
    bits = []
    dc_prev = 0  # DC - główny współczynnik
    for block in blocks:
        dct_block = dct2(block)
        q_block = quantize(dct_block, q)
        zz = zigzag(q_block)
        dc = zz[0]
        ac = zz[1:]
        dc_diff = int(dc - dc_prev)
        dc_prev = dc
        rle_pairs = rle(ac)
        bits.append((dc_diff, rle_pairs))
    return bits


# Dekodowanie JPEG
def jpeg_decode(bits, q=80, img_shape=(512, 512)):
    blocks = []
    dc_prev = 0
    for dc_diff, ac_rle in bits:
        dc = dc_prev + dc_diff
        dc_prev = dc
        ac = irle(ac_rle)
        zz = [dc] + ac
        q_block = inverse_zigzag(np.array(zz))
        dct_block = dequantize(q_block, q)
        block = idct2(dct_block)
        blocks.append(block)
    return block_merge(blocks, img_shape)


# OPCJONALNE 1: (pozniej dodane jescze w pętli w main() )
def get_pil_jpeg_psnr(original_img_path, quality):  # gotowa biblioteka z pythona
    # Wczytaj obraz oryginalny
    original_img_pil = Image.open(original_img_path).convert('L')  # Konwertuj na grayscale dla porównania z algorytmem
    original_img_np = np.array(original_img_pil).astype(np.float32)

    # Zapisz obraz jako JPEG z określoną jakością
    temp_jpeg_path = "temp_pil_jpeg.jpg"
    original_img_pil.save(temp_jpeg_path, quality=quality)

    # Wczytaj skompresowany obraz JPEG
    compressed_img_pil = Image.open(temp_jpeg_path).convert('L')
    compressed_img_np = np.array(compressed_img_pil).astype(np.float32)

    os.remove(temp_jpeg_path)
    return psnr(original_img_np, compressed_img_np)


def main():
    qs = [10, 20, 40, 80, 120]
    psnr_results = {}
    psnr_results_jpeg = {}

    image_files = ["lena512.png", "lena256.png", "barbara512.png", "paski.png"]

    for image_file in image_files:
        path = os.path.join(image_file)
        img = imread(path).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]
        psnrs = []
        psnrs_jpeg = []
        print(f"\nObraz: {image_file}")
        for q in qs:
            # Twój algorytm
            bits = jpeg_encode(img, q)
            out = jpeg_decode(bits, q=q, img_shape=img.shape)
            p = psnr(img, out)
            psnrs.append(p)

            # JPEG z PIL
            p_jpeg = get_pil_jpeg_psnr(path, q)
            psnrs_jpeg.append(p_jpeg)
            print(f"  q={q}: PSNR = {p_jpeg:.2f} dB")

        psnr_results[image_file] = psnrs
        psnr_results_jpeg[image_file] = psnrs_jpeg



    plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--']

    for idx, image_file in enumerate(image_files):
        plt.plot(qs, psnr_results[image_file], marker=markers[idx], linestyle=linestyles[0],
                 label=f"{image_file} (mój algorytm)")
        plt.plot(qs, psnr_results_jpeg[image_file], marker=markers[idx], linestyle=linestyles[1],
                 label=f"{image_file} (JPEG z PIL)")

    plt.xlabel("Wartość q (kwantyzacja)")
    plt.ylabel("PSNR (dB)")
    plt.title("Porównanie PSNR do q – Mój algorytm vs JPEG z PIL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Jeden zbiorczy wykres
    plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--']

    for idx, image_file in enumerate(image_files):
        plt.plot(qs, psnr_results[image_file], marker=markers[idx], linestyle=linestyles[0],
                 label=f"{image_file} (mój algorytm)")
        plt.plot(qs, psnr_results_jpeg[image_file], marker=markers[idx], linestyle=linestyles[1],
                 label=f"{image_file} (JPEG z PIL)")

    plt.xlabel("Wartość q (kwantyzacja)")
    plt.ylabel("PSNR (dB)")
    plt.title("Porównanie PSNR do q – Mój algorytm vs JPEG z PIL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()