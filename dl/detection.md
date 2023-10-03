# Detection Architectures

## RetinaNet Architecture

* The main obstacle to close the gap between two stage detectors and one stage detector is the class imbalance
* In order to overcome this, the proposed solution is a new loss function
* The new loss function is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases.
* A common technique for dealing with class imbalance was hard sampling mining.
* Another technique was to use robust loss functions (Huber loss)

### Architecture

* ResNet Backbone
* FPN Neck
* Regression head and Classification head

The box prediction is made using anchors. The anchors are extracted for every FPN level.

![RetinaNet ARchitecture](imgs/RetinaNetArchitecture.png)

### Loss Function
$$FL(p_t) = - \alpha * (1 - p_t)^{\gamma}\log(p_t)$$

If $p_t$ increases, the loss will decrease. This means that the correctly classified samples are discounted when it comes to loss computation.

