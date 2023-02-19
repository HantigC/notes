# Network in Network

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
MLP, which is a trainable non-linear function aproximators.

Instead of using MLP for projecting the feature maps into the label space, the
current architecture will use global average pooling layes.
