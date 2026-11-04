import numpy as np
import matplotlib.pyplot as plt
import cv2

threshold_value = 127
maximum_value = 255

image = cv2.imread('img/hamster-bw.jpg', 0)

_, binary = cv2.threshold(
    image,
    threshold_value,
    maximum_value,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary image")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("binary.png", binary)