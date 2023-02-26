# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

[[link](http://proceedings.mlr.press/v37/ioffe15.pdf)]

## TLDR
1. Enables using higher learning rates
2. Accelerates training
3. The activation distribution diplays more stability
4. Easier to train sigmoid based activations
5. Easier to train deeper networks
6. Training uses mini-batch statistics, while inference uses whole dataset statistics


## Summary

When the input's distribution changes, the learning system experiences
_covariance shift_. The same phenomenon happens with every intermidiate layer,
meaning that the whole network experiences _internal covariance shift_. Every
layer has to adapt to the change in input (layer's input) distribution.

In order to overcome this issue every leyer is preceded by a batch
normalization (BN) layer. Because of the input's distribution jitter, activations
like sigmoid can experience saturations.

## How it works
The motivation behind DN is that whitening the layers input leads to
an improvement in trainig speed.

BN can be achieved by:
* Using the statistics from the whole dataset.
    > Infeasable if needing to compute it every time a layer makes an weight update
* Whitening using the mini batch statistics
    > Too expensive to compute inverse covariance matrix
* Computing the statistics only for the mini batch.
    > Must include the BN into the differentiation, otherwise it can have no effect. Check the paper for more info

### Computing the statistics only for the mini batch
1. Instead of whitening, only normalize scalar features independetly. $\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$ where $E$ and $Var$ computed over the training dataset.
2. Adding trainable parameters: $\beta^{(k)}$ and $\gamma^{(k)}$. $y^{(k)} = \gamma^{(k)}*\hat{x}^{(k)} + \beta^{(k)}$. These parameters are learned along with the original model parametes.


### Prediction vs Inference
During prediction, the network uses mini batch statistics. During infenrece, the networks uses the statistics from the whole dataset. This results in a linear transformation.

### BN for CNN
Applying BN for CNN differs from FFN. The BN is applyied over all the all kernel locations, resulting in a min-batch size of $\hat{m} = m * q * p$, where $m$ is the batch size, $q$, $p$ being the kernel size.
