# ImageNet Classification with Deep Convolutional Neural Networks

[[link](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)]

## TLDR
1. CNN rchitecture for image classification.
2. CNNs have 11x11, 5x5, 3x3  filters
3. The network is split across two GPUs
4. Achieves best top-1 error (by a high margin)
5. Uses ReLU, Dropout, and channel-wise normalization
6. The vectors extracted from similar images display high cosine similarity

## Architecture

The architecture consists of five Convolutional Neural Networks and three Feed
Forward Networks. Due to the size of the model, which was considered a big one
back in 2012, the training was done by using two GPUs. Some of the layers were
shared between the two GPUs.

### ReLU Nonlinearity

This type of activation solves the saturation problem met when using sigmoid or
tanh. This way the network is able to have non-zero gradients.


### Local Patch Normalization

Altough the ReLU activation doens't require input normalization to prevent them
from saturation, the usage of the local patch normlization aids with reducing
the top-1 and top-5 error on ImageNet.

$$b^i_{x,y} = \frac{a^i_{x,y}}{(k + \Sigma_{j=max(N-1, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a^j_{x,y})^2)^{\beta}}$$

It can be tought as a channel-wise normalization with a window of $n$. $N$ is
the total number of channels in a layer. $k$, $\alpha$ and $\beta$ are some
constants found by using the validation set.

### Overlapping Pooling

The pooling kernel has $stride=2$ and $kernel=3$. This way two adgacent kernel
will overlap.


## Reducing Overfitting
### Data Augmentation
1. Horizontal flips
2. Random 224x224 from a 256x256 image
3. Altering intensities of RBG using sampling along eigen values/vectors.

### Dropout

Using dropout in the first two fully-connected layers.

## Qualitative Evaluations

Similar images result in similar vector representations. Meanig that the cosine
similarity between two vectors will be closer to 1.
