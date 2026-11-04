import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./img/luffy_bw.jpeg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not find input.jpg")

# Binarize image
_, binary = cv2.threshold(
    image,
    127,
    255,
    cv2.THRESH_BINARY
)

# Structuring element
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

# Morphological operations
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

images = [
    image,
    binary,
    eroded,
    dilated,
    opening,
    closing
]

titles = [
    "Original",
    "Binary",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing"
]

plt.figure(figsize=(15, 8))

for index, (current_image, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 4, index + 1)
    plt.imshow(current_image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "morphological_operations.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

cv2.imwrite("eroded.png", eroded)
cv2.imwrite("dilated.png", dilated)
cv2.imwrite("opening.png", opening)
cv2.imwrite("closing.png", closing)