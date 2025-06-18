import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import os


# DCT / IDCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# Kwantyzacja
def quantize(block, q):
    return np.round(block / q)

def dequantize(block, q):
    return block * q


# RGB ↔ YUV
def rgb_to_yuv(img): # Y - jasnosc , U i V - odchylenie koloru od szarosci , po ludzkie oko bardziej odczuwa zmiany jasnosci niz kolory
    img = img.astype(np.float32)
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]])
    return np.tensordot(img, m.T, axes=1) #mnozenie tablic wielowymiarowych(tensorow)

def yuv_to_rgb(img):
    m = np.array([[1, 0, 1.13983],
                  [1, -0.39465, -0.58060],
                  [1, 2.03211, 0]])
    rgb = np.tensordot(img, m.T, axes=1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# PSNR
def psnr(original, reconstructed, b=8):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_i = 2 ** b - 1
    return 10 * np.log10((max_i ** 2) / mse)

def color_psnr(original, reconstructed):
    psnrs = [psnr(original[..., c], reconstructed[..., c]) for c in range(3)]
    return np.mean(psnrs)



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


# Zigzag
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

def inverse_zigzag(input):
    output = np.zeros((8, 8))
    order = np.array(zigzag(np.arange(64).reshape(8, 8)))
    for k, v in enumerate(input):
        idx = np.where(order == k)[0][0]
        i, j = divmod(idx, 8)
        output[i, j] = v
    return output


# RLE
def rle(arr):
    result = []
    zero_count = 0
    for val in arr:
        if val == 0:
            zero_count += 1
        else:
            result.append((zero_count, int(val)))
            zero_count = 0
    result.append((0, 0))
    return result

def irle(pairs):
    result = []
    for zeros, val in pairs:
        if (zeros, val) == (0, 0):
            break
        result.extend([0] * zeros)
        result.append(val)
    while len(result) < 63:
        result.append(0)
    return result


# JPEG encode/decode dla jednego kanału
def jpeg_encode_channel(channel, q):
    blocks = block_split(channel)
    bits = []
    dc_prev = 0
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

def jpeg_decode_channel(bits, q, shape):
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
    return block_merge(blocks, shape)


# JPEG kolorowy
def jpeg_color_encode(img_rgb, q=50):
    img_yuv = rgb_to_yuv(img_rgb)
    h, w, _ = img_yuv.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8

    img_yuv = np.pad(img_yuv, ((0, h_pad), (0, w_pad), (0, 0)), mode='edge')
    components = []

    for c in range(3):
        channel = img_yuv[:, :, c]
        bits = jpeg_encode_channel(channel, q)
        components.append(bits)

    return components, img_yuv.shape

def jpeg_color_decode(components, shape, q=50):
    h, w, _ = shape
    channels = []

    for bits in components:
        channel = jpeg_decode_channel(bits, q, (h, w))
        channels.append(channel)

    img_yuv_rec = np.stack(channels, axis=-1)
    return yuv_to_rgb(img_yuv_rec)


# PSNR porównanie z PIL JPEG
def get_pil_jpeg_psnr(original_img_path, quality):
    original = Image.open(original_img_path).convert('RGB')
    original_np = np.array(original).astype(np.float32)

    temp_path = "temp_pil_color.jpg"
    original.save(temp_path, quality=quality)

    compressed = Image.open(temp_path).convert('RGB')
    compressed_np = np.array(compressed).astype(np.float32)

    os.remove(temp_path)

    return psnr(original_np, compressed_np)


if __name__ == "__main__":
    img_path = 'lena512.png'  # Zmień na nazwę swojego pliku
    q = 50

    # Wczytaj i przekształć obraz
    original_img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)

    # Własna kompresja JPEG
    compressed_bits, shape = jpeg_color_encode(original_img, q)
    decoded_img = jpeg_color_decode(compressed_bits, shape, q)

    # Zapis wyniku
    Image.fromarray(decoded_img).save("decoded_output.jpg")

    # PSNR własny vs PIL
    print("Własny PSNR (kolor):", color_psnr(original_img, decoded_img))
    print("PSNR PIL JPEG:", get_pil_jpeg_psnr(img_path, quality=q))
