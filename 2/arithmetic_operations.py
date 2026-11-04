import cv2
import matplotlib.pyplot as plt

# Load grayscale images
image1 = cv2.imread("./img/cookie.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./img/hamster-bw.jpg", cv2.IMREAD_GRAYSCALE)

if image1 is None:
    raise FileNotFoundError("Could not load image1.jpg")

if image2 is None:
    raise FileNotFoundError("Could not load image2.jpg")

# Make image2 the same size as image1
image2 = cv2.resize(
    image2,
    (image1.shape[1], image1.shape[0])
)

# Arithmetic operations
added = cv2.add(image1, image2)

subtracted = cv2.subtract(image1, image2)

absolute_difference = cv2.absdiff(image1, image2)

blended = cv2.addWeighted(
    image1,
    0.5,
    image2,
    0.5,
    0
)

# Multiplication by a scalar
brighter = cv2.convertScaleAbs(
    image1,
    alpha=1.5,
    beta=0
)

# Addition of a brightness constant
brightness_increased = cv2.add(image1, 50)

# Save results
cv2.imwrite("./img/arith_op/added.png", added)
cv2.imwrite("./img/arith_op/subtracted.png", subtracted)
cv2.imwrite("./img/arith_op/absolute_difference.png", absolute_difference)
cv2.imwrite("./img/arith_op/blended.png", blended)
cv2.imwrite("./img/arith_op/brighter.png", brighter)
cv2.imwrite("./img/arith_op/brightness_increased.png", brightness_increased)

# Display results
images = [
    image1,
    image2,
    added,
    subtracted,
    absolute_difference,
    blended,
    brighter,
    brightness_increased
]

titles = [
    "Image 1",
    "Image 2",
    "Addition",
    "Subtraction",
    "Absolute difference",
    "Weighted blend",
    "Image 1 × 1.5",
    "Image 1 + 50"
]

plt.figure(figsize=(16, 8))

for index, (current_image, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 4, index + 1)
    plt.imshow(current_image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "arithmetic_operations_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

difference = cv2.absdiff(image1, image2)

difference_display = cv2.normalize(
    difference,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

plt.imshow(difference_display, cmap="gray")
plt.title("Absolute difference")
plt.axis("off")
plt.show()