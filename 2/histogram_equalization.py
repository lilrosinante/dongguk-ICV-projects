# Use Histogram Equalization (HE)
# Adaptive Histogram Equalization (AHE)
# Contrast Limited Adaptive Equalization (CLAHE)
# Add a histogram to each of them

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(reference, image):
    """Calculate PSNR in decibels using the original image as reference."""
    reference_float = reference.astype(np.float64)
    image_float = image.astype(np.float64)

    mse = np.mean((reference_float - image_float) ** 2)

    if mse == 0:
        return float("inf")

    max_pixel_value = 255.0
    return 10 * np.log10((max_pixel_value ** 2) / mse)

def plot_histogram(image, label, color):
    """Plot a grayscale histogram."""
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color=color, label=label, linewidth=1.5)

#########################
# Histogram Equalized
img = cv2.imread('img/hamster-bw.jpg', 0)
default_equ = cv2.equalizeHist(img)

#########################
# Adaptive Equalized Histogram
ahe_equ = cv2.createCLAHE(clipLimit=4000.0, tileGridSize=(8, 8)).apply(img)

#########################
# Contrast Adaptive Equalized Histogram
clahe_equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

#########################
plt.figure(figsize=(15, 10))
titles = ["Original", "HE", "AHE", "CLAHE"]
images = [img, default_equ, ahe_equ, clahe_equ]

histogram_colors = [
    "black",
    "red",
    "green",
    "blue"
]

cv2.imwrite("./img/original_grayscale.png", img)
cv2.imwrite("./img/global_he.png", default_equ)
cv2.imwrite("./img/ahe.png", ahe_equ)
cv2.imwrite("./img/clahe.png", clahe_equ)

images = [
    img,
    default_equ,
    ahe_equ,
    clahe_equ
]

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

    plt.subplot(2, 4, i + 5)
    plot_histogram(images[i], titles[i], histogram_colors[i])
    plt.xlim([0, 256])
    plt.ylim(bottom=0)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    plt.grid(alpha=0.25)
    plt.legend()

psnr_original = calculate_psnr(img, img)
psnr_he = calculate_psnr(img, default_equ)
psnr_ahe = calculate_psnr(img, ahe_equ)
psnr_clahe = calculate_psnr(img, clahe_equ)

print(f"Original vs Original: {psnr_original}")
print(f"Original vs Global HE: {psnr_he:.2f} dB")
print(f"Original vs AHE:       {psnr_ahe:.2f} dB")
print(f"Original vs CLAHE:     {psnr_clahe:.2f} dB")

plt.tight_layout()
plt.show()







