"""First exercise: load the real MNIST data and display one image."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

from dataset.mnist import load_mnist


# False preserves the original pixel values from 0 to 255.
# True stores each image as a flat array containing 784 values.
(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True,
    normalize=False,
    one_hot_label=False,
)

print("x_train:", x_train.shape)
print("t_train:", t_train.shape)
print("x_test: ", x_test.shape)
print("t_test: ", t_test.shape)

# Change this index to explore another training image and its label.
index = 0
img = x_train[index]
label = t_train[index]

print("\nActual digit:", label)
print("Before reshape:", img.shape)
img = img.reshape(28, 28)
print("After reshape: ", img.shape)

image = Image.fromarray(np.uint8(img))
output_directory = Path(__file__).resolve().parent / "outputs"
output_directory.mkdir(exist_ok=True)
image.save(output_directory / "first_digit.png")

# The headless option is only needed when checking this script on a server.
if "--no-show" not in sys.argv:
    image.show()


# Inspect the same pixels before and after normalization.
normalized_img = img.astype(np.float32) / 255.0

print("\nOriginal pixel values:")
print(img[10:15, 10:15])

print("\nNormalized values of the same pixels:")
print(np.round(normalized_img[10:15, 10:15], 3))

print("\nOriginal image shape:", img.shape)
print("Normalized image shape:", normalized_img.shape)
