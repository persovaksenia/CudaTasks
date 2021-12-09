#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define n 1000 

__global__ void Pi (double* a)
{
    int i = threadIdx.x;
    a[i] = std::sqrtf(1.0 - double(i * i) / double(n * n));
}

int main()
{
    double a[n];
    double* p_a;

    cudaMalloc((void**)&p_a, n * sizeof(double));
    Pi << <1, n >> > (d_a);

    cudaError_t err = cudaGetLastError();

    cudaMemcpy(a, p_a, n * sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("%s ", cudaGetErrorString(err)); 
    else
    {
        double q = 0;
        for (int i = 0; i < n; ++i) {
            q += a[i];
        }
        printf("pi = %f\n", q * 4 / n);
    }

    cudaFree(p_a);
    return 0;
}