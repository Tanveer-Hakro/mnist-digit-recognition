"""Inspect the remaining MNIST loader options from the supplied page."""

from dataset.mnist import load_mnist


# Keep the image dimensions and store each label as one digit.
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=False,
    flatten=False,
    one_hot_label=False,
)

print('CASE 1: flatten=False, one_hot_label=False')
print('Training image array:', x_train.shape)
print('One image shape:     ', x_train[0].shape)
print('Training label array:', t_train.shape)
print('First actual label:  ', int(t_train[0]))


# Flatten each image and store each label as ten values.
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=False,
    flatten=True,
    one_hot_label=True,
)

print()
print('CASE 2: flatten=True, one_hot_label=True')
print('Training image array:', x_train.shape)
print('One image shape:     ', x_train[0].shape)
print('Training label array:', t_train.shape)
print('First one-hot label: ', t_train[0])
print('The 1 at position 5 represents the actual digit 5.')
