# Source credit and third-party notices

The MNIST loader and trained parameter file come from Koki Saitoh's
**Deep Learning from Scratch** support repository, published by O'Reilly Japan.

- Repository: https://github.com/oreilly-japan/deep-learning-from-scratch
- Loader: `dataset/mnist.py`
- Trained parameters: `ch03/sample_weight.pkl`
- Forward propagation reference: `ch03/neuralnet_mnist.py`
- Original license: `third_party/KOKI_SAITOH_LICENSE.md`

`mnist_predict.py` preserves the original MIT notice for the adapted prediction
calculation. The exact included upstream bytes are recorded with source URLs
and SHA-256 checksums in `third_party/SOURCES.json`.

MNIST images are retrieved by the original loader from its configured mirror:
https://ossci-datasets.s3.amazonaws.com/mnist/

The MNIST figures in `assets/` use actual dataset images and calculated values.
`assets/lesson-guide.png` preserves the guide supplied for this learning exercise;
its illustrated probabilities are distinct from the measured example outputs.
