# Structure from Motion
## [DeepSFM: Structure From Motion Via Deep Bundle Adjustment](https://arxiv.org/pdf/1912.09697.pdf)
## [HSfM: Hybrid Structure-from-Motion](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf)

## [Pixel-Perfect Structure-from-Motion with Featuremetric Refinement](https://arxiv.org/pdf/2108.08291.pdf)
## [FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent](https://arxiv.org/pdf/2404.15259)
![](./imgs/FlowNetForward.png)
### Supervision via Camera-Induced Scene Flow
1. Unproject pixels $\bold{u}_i \in \mathbb{R}^2$ from image $I_{i}$ using depth map $D_{i}$ and intrinsic matrix $K_{i}^{-1}$, yielding a 3D point $x_{i}$
2. Use relative camera pose $P{ij}$ between frames $i$ and $j$ to transform a point $x_{i}$ onto $x_{j}$, yielding an image an implied $\hat{\bold{u}}_{ij}$.
3. Compare the computed correspondence $\hat{\bold{u}}_{ij}$ with a know correspondence $\bold{u}_{ij}$
$$\mathcal{L} = ||\hat{\bold{u}}_{ij} - \bold{u}_{ij}||$$
![](./imgs/FlowLoss.png)
#### Supervision via Dense Optical Flow and Sparse Point Tracks
> Use off-the-shelf point tracker
1. Correspondences derived from two sources: frame by frame optical flow and sparse tracks.
2. Correspondences flow optical flow $\bold{F}_{i}$: $\bold{u}_{ij} = \bold{u}_{i} + \bold{F}_{ij}\left[\bold{u}_{i}\right]$.

### Parameterizing Depth, Pose, and Camera Intrinsics
#### Depth Network.
Use a pre-trained network that estimates depth
#### Pose as a Function of Depth, Intrinsics and Optical Flow
1. Unproject the pixels to create point clouds: $\bold{X}_{j}^{\leftrightarrow}$, $\bold{X}_{i}^{\leftrightarrow}$
2. Use Procrustes formulation to solve the alignment. Use the flow correspondences

$$\bold{P}_{ij} = \text{arg min}_{\bold{P \in SE\left(3\right)}} || \mathcal{W}^{\frac{1}{2}}\left(\bold{X}_{j}^{\leftrightarrow} - \bold{P}\bold{X}_{i}^{{\leftrightarrow}}\right)||$$
![](./imgs/flowMapPose.png)

#### Camera Focal Length as a Function of Depth and Optical Flow.
1. Use multiple candidates $\bold{K}_{k}$
2. Softly select among the 
3. Compute the resulting intrinsics as $\bold{K} = \sum_{k} \omega_{k} \bold{K}_{k}$, $\omega_{k} = \frac{e^{-\mathcal{L}_k}}{\sum_{l}e^{-\mathcal{L}_l}}$

## [Detector-Free Structure from Motion](https://zju3dv.github.io/DetectorFreeSfM/files/main_cvpr.pdf)

## [Global Structure-from-Motion Revisited](https://arxiv.org/pdf/2407.20219)


## [Structure-from-Motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf#page=9&zoom=100,412,202)

### Review of Structure-from-Motion

Usually, the process starts with the feature extraction and matching phase, followed by geometric verification. 
The resulting scene graph -- images as nodes, and the link being the two view dependency between two images -- is used as the foundation for the scene reconstruction. 
In case of incremental SFM, the reconstruction start with initial two view pair selection, continuing with incrementally adding images, then point triangulation, and refining with bundle adjustment

#### Pipeline

![](./imgs/StructureFromMotionRevisited/pipeline.png)

#### Correspondence Search

The first stage is correspondence search. Let $ \mathcal{I} = \left\{  I_i | i = 1 \dots N_{I}\right\}$ the set o images used for reconstruction. 
The output for this stage is a set of geometrically verified pairs of images.  In order to  compute the correspondence graph the pipeline employs:

1. **Feature extraction**: For every image $I_{i}$, the SfM detects sets of $\mathcal{F}_{i} = \left\{\left(\bold{x}_{j}, \bold{f}_{j}\right) | j = 1 \dots N_{F_{i}}\right\}$ at location $ \bold{x}_j \in \mathbb{R}^{2}$ represented by an appearance descriptor $\bold{f}_j$
2. **Feature Matching**: SfM discovers images that see the same scene part by leveraging the features $\mathcal{F}_{i}$. A naive approach would be to compare every image against every image - the complexity in this case is $O\left(N^2_{I}N^2_{F_{i}}\right)$. The output is a set of potentially overlapping image pairs $\mathcal{C} = \left\{\left\{I_{a}, I_{b}\right\} | I_{a}, I_{b} \in \mathcal{I}, a < b\right\}$ and their associated feature correspondences $\mathcal{M}_{ab} \in \mathcal{F}_{a} \times \mathcal{F}_{b}$

3. **Geometric Verification**: In this stage the SfM verifies the potentially overlapping image pairs $\mathcal{C}$. Because the matching relies solely on appearance, it is not guaranteed that the corresponding features actually map to the same point. The SfM pipeline verifies these matches by estimating two-view geometries. Based on the geometries, the features are filtered accordingly. The output of this stage is a scene graph - images being the nodes, the two view pairs being the edges


#### Incremental Reconstruction
The input for the reconstruction stage is the scene graph. The outputs are pose estimates $\mathcal{P} = \left\{\bold{P}_c \in \bold{SE}\left(3\right) | c = 1 \dots N_P\right\}$ for registered images and the reconstructed set of 3d points $\mathcal{X} = \left\{\bold{X}_{k} \in \mathbb{R}^{3} | k = 1 \dots N_{X}\right\}$

1. **Initialization**: SfM initializes the model with a carefully selected image pair. It is benefic for the BA to start from a dense region. Starting from a sparser location may result in bad reconstruction.

2. **Image Registration**: New images can registered to the current model by solving Perspective-n-Point ($\bold{PnP}$) problem. The $\bold{PnP}$ problem involves estimating the camera pose $\bold{P}_{c}$ and the camera intrinsics, in case of uncalibrated camera.

3. **Triangulation**: A newly registered image must observe existing scene points, and it may add new points to the set of points $\mathcal{X}$. A new scene point $\bold{X}_k$  can be added to $\mathcal{X}$ as soon as on more image, also covering the new scene but from different viewpoint, is registered.

4. **Bundle Adjustment**: Having the pose estimation and triangulation as separate procedures results in uncertainties being propagated through the scene. Bundle Adjustment helps with refining the scene (pose and the triangulated points) by minimizing a non-linear projection error: $E = \Sigma_{J} \rho_{j} \left(||\pi\left(\bold{P}_c, \bold{X}_k\right) - x_j||^2\right)$, $\pi$ being the projection function.


#### Challenges


1. The system fails to register large fraction of images
2. Broken models due to mis-registration and drift

These problems might stem from producing an incomplete scene graph or from the failure to register images due to missing or inaccurate scene structure

#### Scene Graph Augmentation

A geometric verification step

1.  Estimate fundamental matrix
2.  If at least $N_F$ are found, consider image pair verified
3.  Estimate homography matrix, $N_H$ being number of homography inliers
4.  moving camera if $\frac{N_H}{N_F} > \epsilon_{HF}$
5.  for calibrated images estimate essential matrix and its number of inliers $N_E$
6.  if $\frac{N_E}{N_F}$ then calibration is correct
7.  In case of correct calibration and $\frac{N_H}{N_F} < \epsilon_{HF}$ then decompose essential matrix
8.  triangulate inlier points from correspondences, and determine median triangulation angle $\alpha_m$
9.  Using $\alpha_m$ to distinguish between pure rotation and planar scenes

#### Next Best View Selection

![](./imgs/StructureFromMotionRevisited/NextImageSelectionScore.png)


The score is based on discretizing the image into $K^{2}_l$ grid. If the 
matches keypoints are uniformly distributed, the score increases.
The score is accumulated over multiple levels with a resolution dependent wights $w_{l} = K^{2}_{l}$

#### Robust and Efficient Triangulation
> Especially for sparsely matched image collections, exploiting transitive correspondences boosts triangulation completeness and accuracy, and hence improves subsequent image registrations

> It is necessary to find a consensus set of track elements before performing a refinement using multiple views.


To handle different levels of outliers the problem is formulated as a triangulation using RANSAC
Let $\mathcal{T} = \left\{T_{n} | n = 1 \dots N_{T}\right\}$ be the features tracks with a priori unknown ratio $\epsilon$ of inliers
A measure $T_{n}$ consists of normalized image observation $\overline{\bold{x}}_{n} \in \mathbb{R}^2$  and the corresponding camera pose $\bold{P}_{n} \in \bold{SE}\left(3\right)$ defining the projection from world to camera frame $\left[R^{T} -R^{T}t\right]$ with $R \in SO\left(3\right)$ and $t \in \mathbb{R}^3$

The objective is to maximize the support of measurements conforming with well-conditioned two-view triangulation
$$X_{ab} \sim \tau\left(\overline{x}_a, \overline{x}_b, P_A, P_b\right) \text{ with } a \not= b$$
where $\tau$ is any chosen triangulation method and $X_{ab}$ is the triangulated point.

Panoramic image pairs are not used for triangulation. A well-conditioned model satisfies two constrains. 

First, a sufficient triangulation angle $\alpha$

$$\cos\alpha = \frac{t_a - X_{ab}}{||t_a - X_{ab}||_2} \frac{t_b - X_{ab}}{||t_b - X{ab}||_2}$$

Second, positive depths $d_a$ and $d_b$ w.r.t. the views $P_a$ and $P_b$, with the depth being defined as $d = \left[p_{31} \space p_{32} \space p_{33} \space p_{34}\right] \left[X^T_{ab} \space 1\right]^T$. A measurement $T_n$ is considered to conform with the moel if it has positive $d_n$ and if its reprojection error 
$$r_n = \left\Vert \overline{x}_n - \left[\begin{matrix}
    \frac{x^{'}}{z^{'}} \\
    \frac{y^{'}}{z^{'}} \\
\end{matrix}\right]\right\Vert_2 \text{ with } 
\left[\begin{matrix}
    x^{'} \\
    y^{'} \\
    z^{'} \\
\end{matrix}\right] = P_n 
\left[\begin{matrix}
    X_{ab} \\
    1
\end{matrix} \right]
 $$ 
 is smaller than a certain threshold $t$

 > RANSAC maximizes K as an iterative approach and generally it uniformly samples the minimal set of size two at random. However, since it is likely to sample the same minimal set multiple times for small NT , we define our random sampler to only generate unique samples. To ensure with confidence η that at least one outlier-free minimal set has been sampled, RANSAC must run for at least K iterations. Since the a priori inlier ratio is unknown, we set it to a small initial value ǫ0 and adapt K whenever we find a larger consensus set (adaptive stopping criterion). Because a feature track may contain multiple independent points, we run this procedure recursively by removing the consensus set from the remaining measurements. The recursion stops if the size of the latest consensus set is smaller than three. The evaluations in Sec. 5 demonstrate increased triangulation completeness at reduced computational cost for the proposed method.

#### Bundle adjustment

* **Parametrization**: There is no need to perform global bundle adjustment after each step, since incremental BA only affects the model locally. So BA is locally performed for most connected components after each image registration 
* **Filtering**: After Ba, some observation do no conform with the model. The observation with large reprojection are filtered. Also, for each point a minimum triangulation angle is enforced. the intrinsic parameters are optimized using BA, excluding principal point - it is an ill posed problem. Cameras with abnormal field of view or a large distortion coefficient magnitude are considered incorrectly estimated and filtered after global **BA**
* **Retriangulation**: pre-BA re-triangulation nad post-BA re-triangulation. the purpose of this step is to improve the completeness of the reconstruction by continuing the track of points that previously failed to triangulate - due to inaccurate poses
* **Iterative Refinement**: Since BA is severely affected by outliers, a second step of BA can significantly improve the results


#### Redundant View Mining
1. efficient camera grouping
2. partition the scene into small, highly overlapping camera groups

BA naturally optimizes mor for the newly extended parts while other parts only improve in case of drift. The unaffected scene parts are grouped as $\mathcal{G} = \left\{G_r | r = 1 \dots N_G\right\}$ - highly overlapping images and each group $G_r$ is counted as a single camera

A image is affected it is newly added or if more than a ration $\epsilon_r$ of its observations have a re-projection error larger than r pixels
> the number of co-visible points between images
is a measure to describe their degree of mutual interaction

For a scene with $N_X$ points, each image can be described by a binar visibility vector $\bold{v}_i \in \left\{0, 1\right\}^{N_X}$, where the $n$-th entry in $\bold{b}_i$ is 1 if point $X_n$ is visible in image i nad 0 otherwise. The degree of interaction between image a and b in calculated using bitwise intersection over union in their vectors $\bold{v}_i$:

$$ V_{ab} = \left\Vert\bold{v}_a \land \bold{v}_b\right\Vert\left / \Vert\bold{v}_a  \lor \bold{v}_b\right\Vert$$

The groups are built as follows:

the images are sorted as $\overline{\mathcal{I}} = \left\{I_i| \left\Vert v_i\right\Vert > \left\Vert v_{i+1}\right\Vert\right\}$
The group is initialized by removing the first image $I_a$  from $\overline{\mathcal{I}}$ and finding the image $I_b$ that maximizes $V_{ab}$

If $V_{ab} > V$ and $\left\vert G_r \right\vert < S$, the image $I_b$ is removed from $\overline{\mathcal{I}}$ and added to group $G_r$

Each image within a group is then parameterized w.r.t a common group-local coordinate frame. the BA cost function for grouped images is

$$E_g = \Sigma_j \rho_j\left(\left\Vert\pi_g\left(G_r, P_c, X_k\right) - x_j\right\Vert_2^2\right)$$ 

extrinsic group parameters $G_r \in SE\left(3\right)$ and fixed $P_c$. The projection matrix of a image in a group is defined as $P_{cr} = P_cG_r$. The overall cost $\overline{E}$ is the sum of the grouped and ungrouped cost contributions

### Experiments

1. **Next Best View Selection**: _"while all strategies converge to the same set of registered images, our method produces the most accurate reconstruction by choosing a better registration order for the images."_
2. **Robust and Efficient Triangulation**: _"Our proposed recursive approaches recover significantly longer tracks and overall more track elements than their non-recursive counterparts"_
3. **Redundant View Mining**: _"The reconstruction quality is comparable for all choices of V > 0.3 and increasingly degrades for a smaller V"_

## [Building Rome in a Day](https://grail.cs.washington.edu/rome/rome_paper.pdf)

Build distributed SfM that reconstruct using internet images

### System Design
![](./imgs/Rome/SystemDesign.png)

#### Preprocessing and feature extraction
The images are available on a central store, from which they are distributed to the cluster nodes on demand. 
1. Verify for EXIF tag
2. Downsample images
3. Extract SIFT features for each image

#### Image Matching

1. Use Ann for matching SIFT features
2. The matches are geometrically verified using RANSAC-based estimation of essential or fundamental matrix.
3. Use object retrieval in order to generate proposals.
   1. vocabulary based
   2. query expansion

##### Vocabulary Tree Proposals
1. Represent image as bag of words
2. Use hierarchical k-means tree to quantize the feature descriptors
3. obtain document frequency and term frequency
##### Verification and detailed matching
1.  > We initially tried to optimize network transfers before any verification is done. In this setup, once the master node has all the image pairs that need to be verified, it builds a graph connecting image pairs which share an image
2.  > The second idea we tried was to over-partition the graph into small pieces, and to parcel them out to the cluster nodes on demand. When a node requests another chunk of work, the piece with the fewest network transfers is assigned to it.
3.  > The approach that gave the best results was to use a simple greedy bin-packing algorithm
    * The master node maintains a list of images of each node
    * When a node asks for work, the master node looks through the image pairs, and adds them if they don't require network transfer
    * A simple solution is to consider only a subset of jobs at a time

##### Merging Connected Components
Let _match graph_ be the graph where the nodes are the images, and the edges being the verified two view geometry. There are two levels of proposals $k_1$ and $k_2$. $k_2$ proposals are used to connect different connected components. The images which do not match any of their $k_1$ proposal, they are discarded.

##### Query Expansion
Using a image as query, instead of retrieving only a document (image) from the db, using query expansion, also the topological neighborhood of that result is retrieved
![](./imgs/Rome/QueryExpansion.png)

