Autor: [Artur D. Nasyrov](https://github.com/Arturawesome)

Laboratory: [Bauman Digital Soft Matter laboratory, BMSTU](http://teratech.ru/en)

Operating System: Manjaro Linux KDE Plasma Version: 5.22.5. 

Processors: 8 × Intel® Core™ i7-9700KF CPU @ 3.60GHz

---

# CUDA Programming Model
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

Функция, используемая для ***передачи данных между хостом и устройством**: cudaMemcpy
```shell
cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,
cudaMemcpyKind kind )
```


# Источники 
PROFESSIONAL CUDA ® C Programmin. John Cheng, Max Grossman, Ty McKercher. Copyright © 2014 by John Wiley & Sons, Inc., Indianapolis, Indiana
