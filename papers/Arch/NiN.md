# Network in Network
## TLDR
* CNN layers act as a GLM, extracting only linear separable features. NiN, on the other hand, extracts nonlinear features.
* GAP (Global Average Pooling) is less prone to overfitting than FFN.
* Channels extracted by GAP act as label likelihood maps.

## Abstract

NiN architecture aims to enhance the model discriminability for local patches
within the receptive fields. It is done by using a sliding feed forward
network. Also, instead of using multiple MLP (multilayer perceptron) at the
end of the network, the current work uses global average pooling layer. It is
less prone to overfitting.

## Introduction

The convolutional filter in CNN acts as a generalized linear model (GLM) for a
specific patch, making the assumption that the patches are located in a linear
separable space. NiN networks aim to overcome this linear assumption by adding
MLP, which is a trainable non-linear function aproximator.

Instead of using MLP for projecting the feature maps into the label space, the
current architecture will use global average pooling layes.

## Network in Network

### MLP Convolutional Layers

Instead of using stacks of of convolutional networks, the NiN architecture uses
MLP convolutional layer. This way it's able to achieve not just linear
discrimination but also non-linear

```math
f^1_{i,j,k_1} = max(w^1_{k_1} * x_{i, j} + b_{k_1}, 0)
\vdots \hbreak
f^n_{i,j,k_n} = max(w^n_{k_n} * f^{n-1}_{i,j} + b_{k_1}, 0)
```


It can be viewed as cascading cross channel MLPs on convolutinal layer.

### Global Average Pooling

It it used as a replacement for the feed forward layers at the end of the
network. It proves to be less prone to overfitting than the FF layers and the
feature maps can be interpreted as confidence maps.
