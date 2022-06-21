Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.24.5

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

GPU: Nvidia GeForce GTX Titan X (3072 CUDA cores)

---

# CUDA Programming Model: Launching a CUDA Kernel
---
We are familiar with the following C/C++ function call syntax:
```shell
function_name (argument list);
```
To call the CUDA kernel, add a kernel’s execution configuration inside triple-angle-brackets:
```shell
kernel_name <<<grid, block>>>(argument list);
```














# Источники 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
