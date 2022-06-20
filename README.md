Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.22.5. 

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

---

# Знакомство с прарллелизмом
---

Сущесвтуют ***два типа парллелизма***: параллелизм задачи и параллелизм данных

***Параллелизм задач*** возникает, когда есть много задач или функций, которые могут выполняться независимо и в значительной степени параллельно. Параллелизм задач фокусируется на распределении функций между несколькими ядрами.

***Параллелизм данных*** возникает, когда имеется много элементов данных, с которыми можно работать одновременно. Параллелизм данных фокусируется на распределении данных между несколькими ядрами.

Парралелизм нужен для:
1. Decrease latency (уменьшения задержки)
2. Increase bandwidth (bandwidth -- пропускная способность данных)
3. Increase throughput (throughput -- пропускная способность операции)

Гетерогенное приложение состоит из двух частей:
1. Код хоста (выполняется на CPU)
2. Код устройства (выполняется на GPU)
Код ЦП отвечает за управление средой, кодом и данными для устройства перед загрузкой ресурсоемких задач на устройстве.
---

# Hello word
---
Самая простая программа реализованная на С и CUDA

```shell
#include <stdio.h>


__global__ void helloFromGPU(void)
{

    printf("Hello from gpu \n");
}
int main()
{
    printf("hello from cpu \n");
    helloFromGPU<<<10, 10>>>(); // <<<N_thread_x N_thread_y>>>
    cudaDeviceReset();
    return 0;
}
```
Компиляция

```shell
$] nvcc hello_workd.cu -o hello
```
Типичная структура программы CUDA состоит из пяти основных шагов:
1. Выделить память графического процессора.
2. Скопируйте данные из памяти CPU в память GPU.
3. Вызовите ядро CUDA для выполнения вычислений, специфичных для программы.
4. Скопируйте данные обратно из памяти GPU в память CPU.
5. Уничтожить память графического процессора.
6. 
В простой программе hello.cu вы видите только третий шаг: вызов ядра.



***[Lammps](https://www.lammps.org/)***  




# Источники 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
