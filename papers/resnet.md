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


