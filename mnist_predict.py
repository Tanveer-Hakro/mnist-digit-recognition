"""Recognize one real MNIST digit using the book's trained neural network.

The prediction calculation follows Koki Saitoh's Chapter 3 example:
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/neuralnet_mnist.py

Original code license:
The MIT License (MIT)
Copyright (c) 2016 Koki Saitoh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.mnist import load_mnist


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a)


def predict(network, x):
    # W contains trained weights; b contains trained biases.
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # Pass the input through the first hidden layer.
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # Pass that result through the second hidden layer.
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # Calculate ten output values, then ten probabilities.
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--index', type=int, default=1, help='MNIST test image index (0 to 9999)')
    parser.add_argument('--no-show', action='store_true', help='Print results without opening an image viewer')
    args = parser.parse_args()

    # Use the input settings expected by these trained parameters.
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True,
        flatten=True,
        one_hot_label=False,
    )
    del x_train, t_train

    if not 0 <= args.index < len(x_test):
        parser.error('--index must be between 0 and 9999')

    root = Path(__file__).resolve().parent
    with open(root / 'assets' / 'sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)

    x = x_test[args.index]
    actual_digit = int(t_test[args.index])
    probabilities = predict(network, x)
    predicted_digit = int(np.argmax(probabilities))

    print('Test image index:', args.index)
    print('Input shape:', x.shape)
    print('Actual digit:', actual_digit)
    print('Output shape:', probabilities.shape)
    print()
    print('Digit   Probability')
    for digit, probability in enumerate(probabilities):
        print(f'{digit:>5}   {probability:.6f}')

    print()
    print('Sum of probabilities:', round(float(probabilities.sum()), 6))
    print('Predicted digit:', predicted_digit)
    print('Highest probability:', round(float(probabilities[predicted_digit]), 6))

    # Restore 0-255 values only for displaying the same input image.
    pixels = np.rint(x.reshape(28, 28) * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    if not args.no_show:
        image.resize((280, 280), Image.Resampling.NEAREST).show()


if __name__ == '__main__':
    main()
