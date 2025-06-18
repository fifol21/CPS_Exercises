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

# Nowe schematy kwantyzacji
def quantize_zone_coding(block, z_coefficients):
    quantized_block = np.zeros_like(block)
    rows, cols = block.shape

    for i in range(rows):
        for j in range(cols):
            if i + j > z_coefficients:
                quantized_block[i, j] = np.round(block[i, j])

    return quantized_block



def quantize_threshold_coding(block, t_threshold):
    return np.where(np.abs(block) < t_threshold, np.round(block), 0)


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
def zigzag(input):
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


def inverse_zigzag(input, block_size=8):
    output = np.zeros((block_size, block_size))
    # Regenerujemy kolejność zig-zag dla danego rozmiaru bloku
    temp_block = np.arange(block_size * block_size).reshape(block_size, block_size)
    order = np.array(zigzag(temp_block))

    # Wypełniamy macierz wartościami z odkodowanego ciągu
    for k, v in enumerate(input):
        if k < len(order):  # Upewniamy się, że nie wyjdziemy poza zakres
            idx = np.where(order == k)[0][0]
            i, j = divmod(idx, block_size)
            output[i, j] = v
    return output


# RLE i odwrotność
def rle(arr):
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


def irle(pairs):
    result = []
    for zeros, val in pairs:
        if (zeros, val) == (0, 0):
            break
        result.extend([0] * zeros)
        result.append(val)
    # Wypełnij resztę bloku zerami do rozmiaru 63 (dla bloku 8x8 AC współczynników)
    while len(result) < 63:  # 64 współczynniki - 1 DC = 63 AC
        result.append(0)
    return result


# Kodowanie JPEG
def jpeg_encode(img, q_value=None, z_coefficients=None, t_threshold=None, block_size=8):
    blocks = block_split(img, block_size)
    bits = []
    dc_prev = 0
    for block in blocks:
        dct_block = dct2(block)

        # Wybór schematu kwantyzacji
        if q_value is not None:
            q_block = quantize(dct_block, q_value)
        elif z_coefficients is not None:
            q_block = quantize_zone_coding(dct_block, z_coefficients)
        elif t_threshold is not None:
            q_block = quantize_threshold_coding(dct_block, t_threshold)
        else:
            raise ValueError("Musisz podać jeden z parametrów kwantyzacji: q_value, z_coefficients lub t_threshold.")

        zz = zigzag(q_block)
        dc = zz[0]
        ac = zz[1:]
        dc_diff = int(dc - dc_prev)
        dc_prev = dc
        rle_pairs = rle(ac)
        bits.append((dc_diff, rle_pairs))
    return bits


# Dekodowanie JPEG
def jpeg_decode(bits, q_value=None, z_coefficients=None, t_threshold=None, img_shape=(512, 512), block_size=8):
    blocks = []
    dc_prev = 0
    for dc_diff, ac_rle in bits:
        dc = dc_prev + dc_diff
        dc_prev = dc
        ac = irle(ac_rle)
        zz = [dc] + ac

        # Tworzymy zrekontruowany blok skwantowanych współczynników
        q_block = inverse_zigzag(np.array(zz), block_size)

        # Odkwantyzacja - musi być spójna ze schematem kwantyzacji
        if q_value is not None:
            dct_block = dequantize(q_block, q_value)
        elif z_coefficients is not None:
            # W przypadku zone-coding i threshold-coding, odkwantyzacja jest prosta
            # bo wartości w q_block są już "odkwantowane" (nie były dzielone przez q)
            # jedynie były zaokrąglane lub zerowane
            dct_block = q_block
        elif t_threshold is not None:
            dct_block = q_block
        else:
            raise ValueError("Musisz podać jeden z parametrów kwantyzacji: q_value, z_coefficients lub t_threshold.")

        block = idct2(dct_block)
        blocks.append(block)
    return block_merge(blocks, img_shape, block_size)


# OPCJONALNE 1:
def get_pil_jpeg_psnr(original_img_path, quality):
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
    image_files = ["lena512.png", "lena256.png", "barbara512.png", "paski.png"]

    # Parametry dla standardowej
    qs = [10, 20, 40, 80, 120]

    # Parametry dla zone-coding
    zs = [1, 3, 5, 8,10]  # Liczba współczynników w lewym górnym rogu (np. 1 dla samego DC, 3 dla 2x2 z pominięciem jednego)

    # Parametry dla threshold-coding
    ts = [5, 10, 20, 50, 100]

    psnr_results_quantize = {}
    psnr_results_zone_coding = {}
    psnr_results_threshold_coding = {}
    psnr_results_jpeg = {}

    for image_file in image_files:
        path = os.path.join(image_file)
        img = imread(path).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]

        print(f"\nObraz: {image_file}")

        # --- Standardowa Kwantyzacja ---
        psnrs_quantize = []
        psnrs_jpeg = []
        print("\n--- Kwantyzacja standardowa (parametr q) ---")
        for q in qs:
            bits = jpeg_encode(img, q_value=q)
            out = jpeg_decode(bits, q_value=q, img_shape=img.shape)
            p = psnr(img, out)
            print(f"  q={q}: PSNR = {p:.2f} dB (Moj algorytm)")
            psnrs_quantize.append(p)

            p_jpeg = get_pil_jpeg_psnr(path, q)
            print(f"  q={q}: PSNR = {p_jpeg:.2f} dB (JPEG z PIL)")
            psnrs_jpeg.append(p_jpeg)
        psnr_results_quantize[image_file] = psnrs_quantize
        psnr_results_jpeg[image_file] = psnrs_jpeg

        plt.figure(figsize=(10, 6))
        plt.plot(qs, psnr_results_quantize[image_file], label="Moj algorytm (standardowa kwantyzacja)", marker='o')
        plt.plot(qs, psnr_results_jpeg[image_file], label="JPEG z PIL", marker='x')
        plt.xlabel(f"Wartość q (kwantyzacja) - {image_file}")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Jakość obrazu w funkcji kwantyzacji (standardowa) - {image_file}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Zone-Coding ---
        psnrs_zone_coding = []
        print("\n--- Kwantyzacja Zone-Coding (parametr z) ---")
        for z in zs:
            bits = jpeg_encode(img, z_coefficients=z)
            out = jpeg_decode(bits, z_coefficients=z, img_shape=img.shape)
            p = psnr(img, out)
            print(f"  z={z}: PSNR = {p:.2f} dB")
            psnrs_zone_coding.append(p)
        psnr_results_zone_coding[image_file] = psnrs_zone_coding

        plt.figure(figsize=(10, 6))
        plt.plot(zs, psnr_results_zone_coding[image_file], label="Moj algorytm (zone-coding)", marker='o',
                 color='green')
        plt.xlabel(f"Liczba współczynników 'z' (zone-coding) - {image_file}")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Jakość obrazu w funkcji 'z' (zone-coding) - {image_file}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Threshold-Coding ---
        psnrs_threshold_coding = []
        print("\n--- Kwantyzacja Threshold-Coding (parametr t) ---")
        for t in ts:
            bits = jpeg_encode(img, t_threshold=t)
            out = jpeg_decode(bits, t_threshold=t, img_shape=img.shape)
            p = psnr(img, out)
            print(f"  t={t}: PSNR = {p:.2f} dB")
            psnrs_threshold_coding.append(p)
        psnr_results_threshold_coding[image_file] = psnrs_threshold_coding

        plt.figure(figsize=(10, 6))
        plt.plot(ts, psnr_results_threshold_coding[image_file], label="Moj algorytm (threshold-coding)", marker='o',
                 color='red')
        plt.xlabel(f"Wartość progowa 't' (threshold-coding) - {image_file}")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Jakość obrazu w funkcji 't' (threshold-coding) - {image_file}")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()


'''
Zachowanie PSNR w zależności od parametru kwantyzacji
Standardowa kwantyzacja (parametr q): PSNR spada wraz ze wzrostem q. Większe q oznacza silniejszą kwantyzację, więcej utraconych informacji i tym samym niższą jakość obrazu.
Zone-coding (parametr z): PSNR rośnie wraz ze wzrostem z. Większe z oznacza zachowanie większej liczby współczynników, co przekłada się na lepszą jakość obrazu (niższe straty).
Threshold-coding (parametr t): PSNR spada wraz ze wzrostem t. Większe t oznacza usunięcie większej liczby współczynników (tylko te z dużą wartością bezwzględną są zachowywane), co prowadzi do większych strat i niższej jakości.
'''