# Deep Residual Learning for Image Recognition

[[arxiv](https://arxiv.org/pdf/1512.03385.pdf)]

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

Instead of aproximating $H\left x \right$, the network should aproximate the
residual mapping $F\left x \right = H\left x \right - x$. As stated in
Introduction, a deeper model shouldn have no greater training error than it's
shallower counterpart. If the identity mappings are optimal, the residual
connection will drive the weights towards 0. Using residual connections it's
easier to find small nudges than to find the whole function (The layer
activation have smaller std Fig. 7). One example could be Fourier Transform.


## identity mapping by shortcuts

The building block is stated as follow:
$$
    y = F \left x , \{ W_i \} + x
$$

$x$ and $y$ being the input and output vectors. $ F \left  x , \{ W_i \} \right
represents residul mapping. The are two cases when it comes to residual mappings.
1. The input and output have the same dimensions. $ y = F \left x , \{ W_i \} + x $
2. The input and output have different dimensions. $ y = F \left x , \{ W_i \} + W_s * x $

The $x$ can be projected in the 1st equation too, but the experiments show that
the identity mapping is enough.
