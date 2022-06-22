 /* key word/fucntion: __global__ kernel, kernel
 * Author: Artur D. Nasyrov
 * Bauman Moscow State Technical University
 * Creating first CUDA Kernel
 */
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
#include <math.h>
#include <chrono>
#include<cuda_runtime.h>

#define CHECK(call)                                                                     \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess)                                                           \
    {                                                                                   \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                                   \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));              \
        exit(1);                                                                        \
    }                                                                                   \
}
void initialData(double *mas, const int N){
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for (int i=0; i<N; i++)
    {
        mas[i] = (double)( rand() & 0xFF )/10.0f;
    }
}
void checkResult(double *gpuRef, double *hostRef, const int N){
    double epsilon = 1.0e-8;
    int match = 1;
    for(int i = 0; i<N; i++)
    {
        if (abs(gpuRef[i] - hostRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void sumArraysOnHost(double *mas1, double *mas2, double *sum, const int N){
    for(int i = 0; i < N; i++)
    {
        sum[i] = mas1[i] + mas2[i];
    }
}

__global__ void sumArraysOnGpu(double *mas1, double *mas2, double *sum){
    //int i = threadIdx.x;
    int i = 16 * (gridDim.x - 1) + threadIdx.x + 4 * threadIdx.y;
    sum[i] = mas1[i] + mas2[i];
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
    "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
    gridDim.x,gridDim.y,gridDim.z);
}

void print_arrays(double *gpu_res, double *h_res, const int nElem){
    for(int i = 0; i<nElem; i++)
    {
         printf("%5.2f;  %5.2f \n", gpu_res[i], h_res[i]);
    }

}


int main(){
    //set up data size of Arrays
    const int nElem = 32;
    //calc requared memory
    size_t req_memory = nElem * sizeof(double);

    // init host data
    double *h_a, *h_b, *h_res, *gpu_res;

    // init device data
    double *d_a, *d_b, *d_r;

    h_a = new double[nElem];
    h_b = new double[nElem];
    h_res = new double[nElem];
    gpu_res = new double[nElem];

    //init of host arrays
    initialData(h_a, nElem);
    initialData(h_b, nElem);

    //put zeros values in results arrays
    memset(h_res, 0, nElem);
    memset(gpu_res, 0, nElem);

    //allocete memory for device data:
    cudaMalloc((double**)&d_a, req_memory);
    cudaMalloc((double**)&d_b, req_memory);
    cudaMalloc((double**)&d_r, req_memory);

    //copy data from host to device
    cudaMemcpy(d_a, h_a, nElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, req_memory, cudaMemcpyHostToDevice);

    //init grid and block size:
    //dim3 block(nElem);
    //dim3 grid(nElem/block.x);
    dim3 block(4, 4, 1);    //block size in threads
    dim3 grid(2, 1, 1);     //grid size in blocks

    // calc our func and kernel
    sumArraysOnHost(h_a, h_b, h_res, nElem);
    sumArraysOnGpu<<<grid, block>>>(d_a, d_b, d_r);

    //copy results from gpu to host:
    cudaMemcpy(gpu_res, d_r, req_memory, cudaMemcpyDeviceToHost);

    //check device and func results:
    checkResult(gpu_res, h_res, nElem);
    print_arrays(gpu_res,h_res, nElem);

    //delete data from gpu
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);

    //delete data from host
    free(h_a);
    free(h_b);
    free(h_res);
    free(gpu_res);

return 0;
}


















































