import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./img/hamster-bw.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not find input.jpg")

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Structuring elements
vertical_line = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.uint8)

horizontal_line = np.array([[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]], dtype=np.uint8)

square_3x3 = np.ones((3, 3), dtype=np.uint8)

cross_3x3 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)

kernels = [
    ("Vertical line", vertical_line),
    ("Horizontal line", horizontal_line),
    ("3x3 square", square_3x3),
    ("3x3 cross", cross_3x3),
]

results = []

for name, kernel in kernels:
    eroded = cv2.erode(
        binary,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    dilated = cv2.dilate(
        binary,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    results.append((name, kernel, eroded, dilated))

# Display
fig, axes = plt.subplots(len(results) + 1, 3, figsize=(12, 4 * (len(results) + 1)))

axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original grayscale")
axes[0, 0].axis("off")

axes[0, 1].imshow(binary, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Binary image")
axes[0, 1].axis("off")

axes[0, 2].axis("off")

for i, (name, kernel, eroded, dilated) in enumerate(results, start=1):
    axes[i, 0].imshow(kernel, cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_title(f"{name} kernel")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(eroded, cmap="gray", vmin=0, vmax=255)
    axes[i, 1].set_title(f"Erosion with {name}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(dilated, cmap="gray", vmin=0, vmax=255)
    axes[i, 2].set_title(f"Dilation with {name}")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("structuring_elements_erosion_dilation.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional save
for name, kernel, eroded, dilated in results:
    safe_name = name.lower().replace(" ", "_")
    cv2.imwrite(f"{safe_name}_eroded.png", eroded)
    cv2.imwrite(f"{safe_name}_dilated.png", dilated)
