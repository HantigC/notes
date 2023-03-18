# Rethinking the Inception Architecture for Computer Vision

[[link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)]

## General design principles

1. Avoid representational bottleneck
    > The representation size should gently decrease
2. Higher dimensional representations are easier to process loclly within a network
3. Spatial aggregation can be done over lower dimensional embeddings without much loss in representation
    > Before applying 3x3, one can reduce the dimension of the input
4. Balance the with and depth of the network

## Factorizing convolutions with Large Filter Size

In vision networks it is expected to have highly correlated nearby activations.
That's why applying dimensional projection doesn't harm the representation.

### Spatial factorization into smaller convolutions

The main idea is to split large convolutions (5x5) into smaller convolutions (3x3)
so that size of the receptive field would be the same. The advantage having smaller
filters is the computational performance.

### Spatial factorization into asymetric convolutions

Following the path of spatial factorization, the convolutions can be factorized
even further. The next type of factorization is spliting two dimentional
convolutions into one dimensional convolutions. This way the  computation cost
is reduced.

According to the experimnets, the asymetric factorization does not work well
on early layer, but it give good results on medium sized grids (m x m feature
maps, $12 \lt m \lt 20$)

## Efficient grid size reduction & InceptionV3

Normally, the pooling operation is applied after a channel expansion. This
expansion is the dominant part of pooling when it comes to the cost of
computation. If the pooling would be applied before the channel expansion then
it would experience representational bottleneck.

The authors propose using in parallel a pooling operation and a convolutional
network, both of them having stride 2.

The inception V3 is constructed by using all the methods described above.

## Label Smoothing

Instead of maximazing the likelihood for a distribution that has the shape of
dirac delta $q(k|x) = \delta_{k, y}$, the authors propose maximazing the
likelihood for a smoothed dirac delta $q(k|x) = \epsilon * \delta_{k, y} +
(1-\epsilon) * u(k) $.

The goal is preventing the largerst logit to becom much larger than the other
ones. This way the model is supposed to generalize better and to be less prone
to overfitting.
