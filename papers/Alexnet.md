# ImageNet Classification with Deep Convolutional Neural Networks

[[link](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)]

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
