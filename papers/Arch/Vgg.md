# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

[[link](https://arxiv.org/pdf/1409.1556.pdf)]

## TLDR
* Deeper netwroks with smaller kernels perform better than shallower networks with bigger kernels
* Training on scale jitter improves the performance
* Multi-scale evaluation improves improves performance
* Multi-crop evaluation improves improves performance

## Next Refs
### Large Scale Distributed Deep Networks
[[link](https://proceedings.neurips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf)]

### OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
[[link](https://arxiv.org/pdf/1312.6229.pdf)]

## Abstract

The main contribution is using 3x3 kernels to increase the depth of the
network to 16-19 layers.

## Introduction
Starting from AlexNet there were different approaches to improve the
architercture. One such improvement refers to usge of smaller kernel size. This
is the strting point of VGG net.

## Architecture
1. 3x3 convolutional kernelsj
2. ReLU nonlinearities
3. Max-pooling with 2x2 layers with stride 2
4. Input is scaled to 224x224
5. Augmentaions consists of subtracting the mean

The usage of 3x3 kernels is motivated by:
1. Multiple non linearities between kernels. This will make the features more discriminative.
2. One 7x7 kernel needs more computaion than three 3x3 kernels in order achieve similar receptive fields.

## Classification Network

### Training
1. batch size = 256
2. momentum = 0.9
3. dropout (0.5) for the two linear layers
4. Reduce on platoeu
5. Smaller netwowrks randomly initialized. Deeper networks initialized from shallower networks.
6. Image size:
    1. Single scale
    2. Jittered multi scale

### Testing

The fully connected layers are replaced by convolutional networks. Namely, the
first fully-connected network is replaced witha 7x7 convolutional layer, while
the remaining two layers are replaces by 1x1 convolutional layers. This output
will result in $c$ receptive channels, $c$ being the number of labels. In order
to get a $c$ dimential vecotr, the final receptive field are mean aggregated.
This way the network is invariate to image size jitters. Alogside the previous
transformation, the images are flipped horizotally, the score being average
between the two.

## Evaluations

### Single Scale Evalution
1. Classification error decreases with the increase in depth
2. 1x1 performs worse than 3x3
3. Deeper networks with smaller kernel perform better than shllower networks with bigger kernels
4. Training on scale jitter leads to significatly better results than fixed scale

### Multi-Scale Evalution
Multi-scale evaluation results in better performance. This statement applies
regardless of training strtegy (single vs multi scale).

### Multi-Crop Evalution
Multi-crop evaluatio perform better than dense evaluation.
