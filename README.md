# neural-net

Live demo of a handwritten digit recognizer, using a network trained with this software:
https://numbernet.netlify.app/

Sister repo for the above demo: https://github.com/use/digit-recognizer

This is a neural network trainer. The main software trains on the MNIST dataset - a collection of images + correct labels for 60,000 handwritten digits + 10,000 test cases. A CSV version of MNIST was obtained from https://www.kaggle.com/oddrationale/mnist-in-csv

The software can train using the CPU in online mode, or the GPU in batch mode. The GPU implementation is currently much slower.

The network can be configured to run with multiple layers but only one seems to be useful, with 20 neurons providing fast training plus good accuracy of around 94%.

## Requirements

  * Nvidia CUDA-capable video card
  * CUDA libraries and nvcc compiler
  * Get some data from https://www.kaggle.com/oddrationale/mnist-in-csv and put it in ./data/

## Compile

```make clean && make```

## Run

Argument should be either "gpu" or "cpu" to run that type of training.
  -n <number> Use <number> neurons in the hidden layer (default 20)
  -t <number> Use <number> training samples (default 10000)
  -v <number> Use <number> verification (testing) samples (default 1000)
  -e <number> Run <number> epochs (default 1)
  -l <number> Learning rate (default 0.050000)
  -b <number> Use <number> batch size (default 64) ("gpu" only)
Program defaults:
  Hidden Layer Neurons: 20
  Training Samples: 10000
  Test Cases: 1000
  Epochs: 1
  Learning Rate: 0.050000
  Batch Size: 64

## Example Command + Output
```
$ ./project -n 20 -t 10000 -v 1000 -e 1 -l .1 cpu
Hidden Layer Neurons: 20
Training Samples: 10000
Test Cases: 1000
Epochs: 1
Learning Rate: 0.100000
Training with CPU
Initialized weights
Got training data
sample 0
sample 1000
sample 2000
sample 3000
sample 4000
sample 5000
sample 6000
sample 7000
sample 8000
sample 9000
Done training epoch 0
Accuracy: 0.82
```
