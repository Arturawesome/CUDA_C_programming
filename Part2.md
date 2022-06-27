Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.22.5. 

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

---

# CUDA Programming Model: Organizing Threads
---
![](https://github.com/Arturawesome/CUDA_C_programming/blob/main/figures/CUDA_fig_2_1.png)

All threads spawned by a single kernel launch are collectively called a grid. All threads in a grid share the same global memory space. A grid is made up of many thread blocks. A thread block is a group of threads that can cooperate with each other using:
1. Block-local synchronization
2. Block-local shared memory

__Threads from different blocks cannot cooperate.__

Обычно сетка организована как двумерный массив блоков, а блок организован как трехмерный массив потоков. И сетки, и блоки используют тип dim3 с тремя целочисленными полями без знака. Неиспользуемые поля будут инициализированы до 1 и проигнорированы.

CUDA организует сетки и блоки в трех измерениях. На рис. 2-5 показан пример структуры иерархии потоков с двумерной сеткой, содержащей двумерные блоки. Размеры сетки и блока задаются следующими двумя встроенными переменными:
```shell
1. blockDim (размер блока, измеренный в потоках)
2. gridDim (размер сетки, измеренный в блоках)
```
Эти переменные имеют тип ***dim3*** , целочисленный векторный тип, основанный на uint3, который используется для указания размеров. При определении переменной типа dim3 любой неуказанный компонент инициализируется значением 1. Каждый компонент в переменной типа dim3 доступен через его поля x, y и z соответственно, как показано в следующем примере:

The coordinate variable is of type uint3, a CUDA built-in vector type, derived from the basic inte-
ger type. It is a structure containing three unsigned integers, and the 1st, 2nd, and 3rd components
are accessible through the fi elds x, y, and z respectively.
```CUDA
blockIdx.x
blockIdx.y
blockIdx.z

threadIdx.x
threadIdx.y
threadIdx.z
```

The code example:
```CUDA
 /* key word/fucntion: dim3 block, grid
 * Author: Artur D. Nasyrov
 * Bauman Moscow State Technical University
 * Inroduction in size of grids and boxes
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

int main()
{
    //define tital number of element
    int nElem = 1024;
    // задание одномерной сетки
    dim3 block(1024);  // размер блока в количестве потоков на один блок
    dim3 grid((nElem + block.x - 1) / block.x);  // размер СЕТКИ в количестве блоков на сетке (округление всегда в меньшую сторону)
    printf("grid.x %d block.x %d \n",grid.x, block.x);

    // reset block
    block.x = 512; // размер блока в 512 потоков
    grid.x = (nElem + block.x -1) / block.x; // размер сетки в 2 блока
    printf("grid.x %d block.x %d \n",grid.x, block.x);

    block.x = 256; // block size in 256 threads
    grid.x = (nElem + block.x -1 ) / block.x; //grid size in 4 blocks
    printf("grid.x %d block.x %d \n",grid.x, block.x);

    // block size in 1024 thread in 3D
    block.x = 64;
    block.y = 4;
    block.z = 4;
    grid.x = 1;
    printf("grid (%d, %d, %d);  block (%d, %d, %d) \n",grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // block size in 1024 threads in 3D (32, 4, 2) = 256 threads
    // grid size in 4 boxes
    block.x = 32;
    block.y = 4;
    block.z = 2;
    grid.x = (nElem + block.x - 1) / (block.x * block.y * block.z * 2);
    grid.y = (nElem + block.y - 1) / (block.x * block.y * block.z * 2);
    printf("grid (%d, %d, %d);  block (%d, %d, %d) \n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
    cudaDeviceReset();
return 0;
}
```

# Источники 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
