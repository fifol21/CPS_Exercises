import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, dctn, idctn

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho') # wykonujemy DCT w pionie a potem w poziomie ( nasze 2D)

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def show_image(img, title="", cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1
img1 = cv2.imread("im1.png", cv2.IMREAD_GRAYSCALE)
img1 = img1.astype(float)
dct_img1 = dct2(img1)
show_image(np.log(np.abs(dct_img1)+1), "DCT2 of im1.png")


flat = np.abs(dct_img1).flatten()    # splaszczamy maicerz i uklady je malejaco
sorted_indices = np.argsort(flat)[::-1]
keep = sorted_indices[:len(flat)//2] # zachowywujemy tylko polowe
mask = np.zeros_like(dct_img1)
mask_flat = mask.flatten()# tworzymy macierz na te co chcemy usunac
mask_flat[keep] = 1
mask = mask_flat.reshape(dct_img1.shape)
dct_reduced = dct_img1 * mask   # mnozenie macierzowe - zostaje tylko to co chcemy
img1_reconstructed = idct2(dct_reduced)
show_image(img1_reconstructed, "Reconstructed im1.png (50% DCT)")

# 2
img2 = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
img2 = img2.astype(float)
dct_img2 = dct2(img2)
show_image(img2, "Original cameraman.tif")
show_image(np.log(np.abs(dct_img2)+1), "DCT2 of cameraman")  # wczytanie obrazu i pokazanie jego DCT

# a) Wyzeruj współczynniki wysokich częstotliwości
threshold_freq = 50
mask = np.zeros_like(dct_img2)
mask[:threshold_freq, :threshold_freq] = 1 # tworzymy maske ktora usunie nam wszytskie wysokie czestotwliosci i zostawi tylko niskie - 1 tam zostaje reszta 0
dct_lowpass = dct_img2 * mask
img_lowpass = idct2(dct_lowpass)
show_image(img_lowpass, "Low Frequencies Only")  # rekonstrukcja naszego obrazu - wniosek energia skupiona jest na lowpass

# b) Zostaw tylko te, które przekraczają próg wartości bezwzględnej
threshold = 50
mask = np.abs(dct_img2) > threshold
dct_thresh = dct_img2 * mask
img_thresh = idct2(dct_thresh)
show_image(img_thresh, "Values > threshold (abs)") # pokazuje nam to ze wyzerownaie wartosci ponizej progu, dalej pozwala nam to na odtworzenie informacji chodz z utrata szegolow, wiele wspolczynnikow mozna usunac bez utraty danych

# 3
size = 128
IM1 = np.zeros((size, size))   # tworzymy macierze DCT z jednym niezerowym wspolczynnikiem, oznaczaja one konkretne f w pionie i poziomie
IM2 = np.zeros_like(IM1)
IM3 = np.zeros_like(IM1)

IM1[2, 10] = 1
IM2[5, 3] = 1
IM3[7, 7] = 1

im1 = idct2(IM1)
im2 = idct2(IM2)
im3 = idct2(IM3)

im = im1 + im2 + im3
show_image(im1, "Base Image 1")
show_image(im2, "Base Image 2")
show_image(im3, "Base Image 3")
show_image(im, "Combined Image")

dct_im = dct2(im)
show_image(np.abs(dct_im), "DCT of Combined")

# Zachowaj tylko jeden współczynnik i odtwórz jeden obraz bazowy
only_one = np.zeros_like(dct_im)
only_one[2, 10] = dct_im[2, 10]
reconstructed = idct2(only_one)
show_image(reconstructed, "Single Basis Image (2,10)", cmap='viridis') # Pokazuje to, że DCT rozkłada obraz na sumę takich bazowych wzorów i każdy współczynnik określa udział danej częstotliwości w obrazie.

# 4
img_zagadka = cv2.imread("im2.png", cv2.IMREAD_GRAYSCALE)
dct_zagadka = dct2(img_zagadka.astype(float))
show_image((np.abs(dct_zagadka)), "DCT2 of Zagadka") # Moj wniosek jest taki ze niskie czestotwliosci maja zdecydoanie wiecej energii, i zostaje ona zagloszona i nie widac wysokich czestotwlisoci, to pokazuje efektywnosc kompresji za pomoca DCT