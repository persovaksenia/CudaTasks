#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 1000 

__global__ void DzetaFunction(float* a, float* b)
{
    int i = threadIdx.x;
    a[i] = 1.f / powf(float(i + 1), *b);
}


int main()
{
    float s = 2;
    float a[N]; 
    float* dev_s = 0;
    float* dev_a = 0;
    float sum = 0; 

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_s, sizeof(float));
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s, &s, sizeof(float), cudaMemcpyHostToDevice);

    ZFunction << <1, N >> > (dev_a, dev_s);

    cudaMemcpy(a, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        sum += a[i];
    }
    printf("%f\n", sum);

    cudaFree(dev_a);
    cudaFree(dev_s);
    return 0;
}