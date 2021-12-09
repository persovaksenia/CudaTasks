#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 10 

__global__ void Calculate (float* a, float* b, float* c)
{
	int i = threadIdx.x; //индексирование
	if (i > N - 1) return; 	//проверка на выход за пределы массива
	//поэлементное умножение массивов
	c[i] = __fmul_rn(a[i], b[i]);
}

int main()
{
	// выделение памяти под массивы на CPU
	float a[N], b[N], c[N];
	// выделение памяти под массивы для копирования
	// на GPU
	float* dev_a, * dev_b, * dev_c;

	// заполнение массивов
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
		b[i] = -2;
	}

	// выделение памяти под массивы на GPU
	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_c, N * sizeof(float));

	// копирование данных в память GPU
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	Calculate << <1, N >> > (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	float result = 0; //переменная для результата

	//подсчёт скалярного произведения
	for (int i = 0; i < N; i++)
	{
		result += c[i];
	}

	//вывод результата
	printf("result = %f\n", result);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}