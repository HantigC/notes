# Deep Residual Learning for Image Recognition

[[arxiv](https://arxiv.org/pdf/1512.03385.pdf)]

## Main Points
1. Plain networks display degradation - deeper networks have bigger trainig error
2. It's easier to learn residual function than full functions -- Can be thought as a Fourier
3. The layer activations have samller stndard deviations.
4. Projection doesn't bring substantial increase in performace.



## Abstract

* Leads to 8x depper than [[VGG](vgg.md)].
* 1st place on ILSVRC 2015 classification task.

## Introduction

Depth is of crucial importance. However, deeper networks can't be achieved by
simply stacking multiple layers, this can lead to vanishig/exploding gradients,
which hampers the convergence. Another problem which occurs when training
deeper networks is accuracy saturation and degradation. This degradation is not
caused by overfitting.

### Idea

* One solution is copying the shallower network, and then adding identity mappings
* The depper networks should perform worse than their shallower architectures.
* The paper tries to solve this by adding _deep residual networks_.
* Instead of fitting the whole mappig, the stacked layers will fit residual mappings.
    > The hypothesis is that it's easier to fit optimize the residual mapping
    > than to optimize the original, unreferenced mapping.

## Related Work

1. Residual Representation
    * Problems solved by optimizing residuals
2. Shortcut Connections
    * Input connected to output
    * Highway networks (gated functions)


## Deep Residual Learning

Instead of aproximating $H(x)$, the network should aproximate the
residual mapping $F(x) = H(x) - x$. As stated in
introduction, a deeper model shouldn have no greater training error than it's
shallower counterpart. If the identity mappings are optimal, the residual
connection will drive the weights towards 0. Using residual connections it's
easier to find small nudges than to find the whole function (The layer
activation have smaller std Fig. 7). One example could be Fourier Transform.


### identity mapping by shortcuts

The building block is stated as follow:
$$y = F (x, W_i) + x$$

$x$ and $y$ being the input and output vectors. $F(x, \{W_i\})$
represents residul mapping. The are two cases when it comes to residual mappings.
1. The input and output have the same dimensions. $y = F(x, \{ W_i \}) + x$
2. The input and output have different dimensions. $y = F(x, \{ W_i \}) + W_s * x$

The $x$ can be projected in the 1st equation too, but the experiments show that
the identity mapping is enough.

### Network Architecures

Comparison between _Plain Network_ and _Residual Network_. Plain network is
inspired by the philosiphy of VGG nets. Residul network is based on plain
network, the difference being the residul connection afert two convolutional
blocks.

### Implementation

1. Image resized with its shorter side randomly sampled from [256, 480]
2. 224x224 random crop from image or its horizontal flip
3. Per-pixel mean subtracted.
4. Batch Normalization after each convolution and before each activation.
5. SGD with mini-batch of 256.
6. Lr starts from 0.1 and is divided by 10 when the error plateaus.
7. No dropout.
8. 10-crop testing

## Experiments

### Plain vs Residual
Two versions on networks are employed: 18-layer and 34-layer. Residual networks
display a better performance when compared to the plain networks.
The 34-layer plain network displays degradation.

### Identity vs Projection Shortcuts

The are employed three types of shortcut connections:
1. Zero padding for increase in dimension.
2. Projection for increase in dimension, the rest are identity.
3. All shortcuts are projection.

Second approach is slightly better than first, the third slightly better than
second.


### Bootleneck Architecures

The pair of two 3x3 convolution is replaced by [1x1, 3x3, 1x1] stack of connections.
This will result in reducing the dimension and then restoring it, and at the end
a residual connection.

### Analysis of Layer Response

By using empirical methods, it is shown that the standard deviations of the
layer responses are more stable and smaller. This correlates with the hypothesis
stated in Introduction.

### Exploration of 1000 layers

The experiment showed that the network keeps imporving, being able to achienve
a training error smaller tahn 0.1%, also being able to mantaing a good test error.
