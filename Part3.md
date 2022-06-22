Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.24.5

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

GPU: Nvidia GeForce GTX Titan X (3072 CUDA cores)

---

# CUDA Programming Model: Launching a CUDA Kernel
---
We are familiar with the following C/C++ function call syntax:
```C++
function_name (argument list);
```
To call the CUDA kernel, add a kernel’s execution configuration inside triple-angle-brackets:
```CUDA
kernel_name <<<grid, block>>>(argument list);
```
***grid*** - grid size in number of block
***block*** - block size in number of threads

The first value in the execution configuration is the grid dimension, the number of blocks to launch. The second value is the block dimension, the number of threads within each block. By specifying the grid and block dimensions, you configure:
1. The total number of threads for a kernel
2. The layout of the threads you want to employ for a kernel (схема потоков для ядра, которое вы хотите испольщовать)

![](https://github.com/Arturawesome/CUDA_C_programming/blob/main/figures/CUDA_fig_2_6.png)

For example, suppose you have 32 data elements for a calculation.You can group 8 elements into each block, and launch 4 blocks, or group all 32 elements into one block or each block just have one element, you have 32 blocks as follows:
```CUDA
kernel_name <<<4, 8>>>(argument list);
kernel_name <<<1, 32>>>(argument list);
kernel_name <<<32, 1>>>(argument list);
```
A kernel call is asynchronous with respect (по отношению) to the host thread. After a kernel is invoked, control returns to the host side immediately (немедленно). You can call the following function to force the host application to wait for all kernels to complete.
```shell
cudaError_t cudaDeviceSynchronize(void);
```
Some CUDA runtime APIs perform an implicit synchronization between the host and the device. When you use ***cudaMemcpy*** to copy data between the host and device, implicit synchronization at the host side is performed and the host application must wait for the data copy to complete.
```CUDA
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```
> Unlike a C/C++ function call, all CUDA kernel launches are asynchronous. Control returns to the CPU immediately after the CUDA kernel is invoked.

Type of cernel in CUDA you can see in table 2_2:
![](https://github.com/Arturawesome/CUDA_C_programming/blob/main/figures/CUDA_Table_2_2.png)

The code example of realization of simple CUDA cernel and C++ function respectivelly:
```CUDA
__global__ void sumArraysOnGPU(double *mas1, double *mas2, double *sum)
{
    int i = threadIdx.x;
    sum[i] = mas1[i] + mas2[i];
}
void sumArraysOnHost(double *mas1, double *mas2, double *sum, const int N)
{
    for(int i = 0; i < N; i++)
    {
        sum[i] = mas1[i] + mas2[i];
    }
}
```
You’ll notice that the loop (цикл) is missing (отсутствует), the built-in thread coordinate variables are used to replace the array index, and there is no reference to N as it is implicitly defined by only launching N threads(N нявно определяется при запуске ядра). Supposing a vector with the length of 32 elements, you can invoke the kernel with 32 threads as follows:
```CUDA
sumArraysOnGPU<<<1,32>>>(float *A, float *B, float *C);
```
 ***[See suplimentarty](https://github.com/Arturawesome/CUDA_C_programming/blob/main/Programs/prog_2_4.cu)***



# Bibliography 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
