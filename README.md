# hpc_lecture

Student ID: 20M38245
Name: Yang Lei


|  Method         |  GLops   |  Time(s) |  error   |
| --------------- | -------- | ---------| -------- |
| OpenMP          | 0.621291 | 0.054008 | 0.000016 |
| SIMD            | 5.712762 | 0.005874 | 0.000016 |
| cacheblocking   | 0.534946 | 0.062725 | 0.000012 |
| combination     | 3.105487 | 0.010805 | 0.000012 |
| CUDA            | 7.265202 | 0.004619 | 0.000012 |

The combination is using OpenMP, SIMD and MPI to accelerate the calculation together.
The best acceleration method is CUDA based on GPU.
