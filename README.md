# CUDA_Samples

These are sample source codes explained in megayuchi's programming course. 

## [Requirements] 
1. Windows 10 build 19041 or later 
2. Visual Studio 2019 
3. CUDA SDK 11.4
--------------------------------------------------------------------------------

## [samples]
1. cudaTransformFloat4 - Multiply the 4x4 matrix by about 100 million points. 
2. CUDA_SelectValue - Find the maximum of each set of numbers.
    Compare the following three cases.
    1.CPU single thread
    2.CUDA - don't using shared memory,
    3.CUDA - using shared memory 

--------------------------------------------------------------------------------

## [Run & Test]
1. Debug builds of CUDA code are incredibly very very. Use the release build for normal speed comparison. 
2. Debugging with nvidia NSight is very very very slower than debug builds. Use NSight debugging only when absolutely necessary. 
