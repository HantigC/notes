# Going deeper with convolutions

[[link](https://arxiv.org/pdf/1409.4842.pdf)]

## Sumarry

The most straightforward way of improving the performance of a deep neural
network is by increasing its depth (number of levels) or width (number of units
at each level).

Even though being a simple solution, it comes with two major drawbacks. Bigger
networks means larger number of trainable parameters, which leads to
overfitting if there is not enough training data. The second draback is that
bigger networks mean more computational resources (increase in kernel size
results in quadratic increase in computation).

One way to overcome these issues would be moving from fully connected
architectures to sparsely connected architecures. According to Aurora et al, if
a distribution of a dataset can be aproximated by a large and sparse network
then it can be contructed by analising the correlation of the activation of the
last layer and clustering neurons with highly correlated autputs - Hebian
principle.

### Architecture

Ussing the idea form Aurora et al correlated outputs should be clustered into
groups, and then these groups are connected to units of previous layers.

The authors of Inception assume these units correspond to regions in the input
image and they are grouped into filter banks. Using this reasoning it means
that a lot of clusters will be concentrated in a single region and it will be
possible to capture them using 1x1 convolutions. Besides these alligned
clusters there will be some spread out clusters. In order to capture these
aswell the authors propose using also 3x3 and 5x5 convolutions + max pooling
alligned horizontaly.

One problem that arises is computational cost when it comes to 5x5
convolutions. Before applying 3x3 and 5x5 convolutions, a 1x1 convolution will
be applied in order to project the number of channels.

This horizontaly alligned 1x1,3x3,5x5, maxpool and projections will be called
and inception block. The blocks will be repeted multiple times but only at the
end of the network, the beginning of the network will be a stack of
convolutions.
