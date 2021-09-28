#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>
#include <vector>


__global__ void MetodMonteKarlo(int* maxPointsCount, int* pointsInCircleCount, double* coordinateX, double* coordinateY)
{
    for (int i = 0; i < *maxPointsCount; ++i)
    {
        double z = (coordinateX[i] * coordinateX[i]) + (coordinateY[i] * coordinateY[i]);

        if (z <= 1)
        {
            ++* pointsInCircleCount;
        }
    }
}

int main()
{
    //переменные на CPU
    const int maxPointsCount = 15000;
    int pointsInCircleCount = 0;
    double pi;
    //переменные на GPU
    int * dev_maxPointsCount, * dev_pointsInCircleCount;
    int size = sizeof(int); // размерность
    double coordinateX[maxPointsCount], * dev_x;
    double coordinateY[maxPointsCount], * dev_y;
    const size_t x_size = sizeof(double) * size_t(maxPointsCount);

    srand(time(NULL));

    for (int i = 0; i < maxPointsCount; i++)
    {
        coordinateX[i] = (double)rand() / RAND_MAX;
        coordinateY[i] = (double)rand() / RAND_MAX;
    }
    // выделение памяти на GPU
    cudaMalloc((void**)&dev_maxPointsCount, size);
    cudaMalloc((void**)&dev_pointsInCircleCount, size);

    cudaMalloc((void**)&dev_x, x_size);
    cudaMalloc((void**)&dev_y, x_size);

    //копирование информации с CPU на GPU
    cudaMemcpy(dev_maxPointsCount, &maxPointsCount, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointsInCircleCount, &pointsInCircleCount, size, cudaMemcpyHostToDevice);

    cudaMemcpy(dev_x, coordinateX, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, coordinateY, x_size, cudaMemcpyHostToDevice);

    // вызов ядра
    MetodMonteKarlo << < 1, 1 >> > (dev_maxPointsCount, dev_pointsInCircleCount, dev_x, dev_y);

    // копирование результата работы ядра с GPU на CPU
    cudaMemcpy(&pointsInCircleCount, dev_pointsInCircleCount, size, cudaMemcpyDeviceToHost);

    pi = ((double)pointsInCircleCount / (double)maxPointsCount) * 4.0;
    printf("Pi: %f\n", pi);

    return 0;
}

