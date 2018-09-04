Requirements: 
 - Gflags, Eigen, Blas and Lapack, Ceres, Cuda, Magma dense

Optional:
 - Glog, Matlab, OpenMVG, SuiteSparse

-----------------------------------------------------

Installation: 
1) Precompile/download required libraries
2) Build the project using Cmake

-----------------------------------------------------

Output:
It can be saved txt file with upper triangles of symmetric covariance matrices, i.e. the values 1-6 for each point covariance matrix 
-----------
| 1  2  3 |
| 2  4  5 |
| 3  5  6 |
----------- 