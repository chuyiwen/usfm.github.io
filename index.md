## Welcome to Uncertainty4SfM Pages

This repository contains the original source codes used for ECCV2018 publication: [Fast and Accurate Camera Covariance Computation for Large 3D Reconstruction](http://people.ciirc.cvut.cz/~policmic/files/Polic_Fast_and_Accurate_Camera_Covariance.pdf)

I recomend to use the new extended version which will be published soon.

### External libraries
Requirements: 
 - Gflags, Eigen, Blas and Lapack, Ceres, Cuda, Magma dense
Optional:
 - Glog, Matlab, OpenMVG, SuiteSparse

### Installation 
1) Precompile/download required libraries
2) Build the project using Cmake

### Output
It can be saved txt file with upper triangles of symmetric covariance matrices, i.e. the values 1-6 for each point covariance matrix 
-----------
| 1  2  3 |
| 2  4  5 |
| 3  5  6 |
----------- 

### Known issues
- The codes work with one to one camera-image corespondences. It does not assume shared intrinsics.
- It works with PINHOLE, SIMPLE_RADIAL and RADIAL disortion model
- It didn't compute the uncertainties of points in 3D
(All the issues are solved now in new version of software which will be published during October 2018.)

### Support or Contact

Detailed information may be on my [web page](http://people.ciirc.cvut.cz/~policmic).
If you have any questions please contact me and I’ll help you sort it out.
