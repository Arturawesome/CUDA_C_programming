Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.22.5. 

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

---

# CUDA Programming Model: allocate memory
---
Типичный поток обработки программы CUDA следует следующему шаблону:
1. Скопируйте данные из памяти CPU в память GPU.
2. Вызывать ядра для работы с данными, хранящимися в памяти графического процессора.
3. Скопируйте данные обратно из памяти GPU в память CPU.

Функция, используемая для ***выделения памяти графического процессора***, называется cudaMalloc:

```shell
cudaError_t cudaMalloc ( void** devPtr, size_t size )
```
Эта функция выделяет линейный диапазон памяти устройства с заданным размером в байтах. Выделенная память возвращается через devPtr. Можно заметить поразительное сходство между cudaMalloc и стандартной библиотекой времени выполнения C malloc. Это сделано намеренно. Сохраняя интерфейс как можно ближе к стандартным библиотекам времени выполнения C, CUDA упрощает перенос приложений.

Функция, используемая для ***передачи данных между хостом и устройством***: cudaMemcpy
```shell
cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
```
Эта функция копирует указанные байты из исходной области памяти, на которую указывает src, в целевую область памяти, на которую указывает dst, с направлением, указанным видом, где вид принимает один из следующих типов.
1. cudaMemcpyHostToHost
2. cudaMemcpyHostToDevice
3. cudaMemcpyDeviceToHost
4. cudaMemcpyDeviceToDevice

# Example of code
```CUDA
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>



void sumArrayOnHost(float *a, float *b, float *c, const int N){
    for (int i = 0; i < N; i ++)
    {
        c[i] = a[i] + b[i];
    }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

}
int main(){
    int nElem = 1024;
    float *h_a, *h_b, *h_c, *gpuRef;     //memory for CPU (host)
    float *d_a, *d_b, *d_c;     //memory for GPU (device)
    h_a = new float[nElem];
    h_b = new float[nElem];
    h_c = new float[nElem];
    gpuRef = new float[nElem];

    initialData(h_a, nElem);
    initialData(h_b, nElem);

    cudaMalloc((float**)&d_a,  nElem * sizeof(float));      //allocate memory on the GPU for array d_a
    cudaMalloc((float**)&d_b,  nElem * sizeof(float));      //выделение памяти на гпу для массива d_b
    cudaMalloc((float**)&d_c,  nElem * sizeof(float));

    cudaMemcpy(d_a, h_a, nElem * sizeof(float), cudaMemcpyHostToDevice);   // копирование массива h_a в массив d_a их оста к девайсу цпу - гпу)
    cudaMemcpy(d_b, h_b, nElem * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpuRef, d_c, nElem * sizeof(float), cudaMemcpyDeviceToHost);

    sumArrayOnHost(h_a, h_b, h_c, nElem);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);
    cudaFree(d_b);





    return(0);
}

```
# Источники 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
