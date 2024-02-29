# llama2 on milkv duo

## llama2.c 版本
### 1.Q8_0量化

`$ python export.py stories15M.q80 --version 2 --checkpoint ./stories15M.pt `
量化权重结果: **stories15M.q80**。

### 2.编译二进制

`$ make run`
make结果: **run-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead**、**runq-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead**，其中**runq-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead**为量化版本二进制文件。

### 3.编译镜像
使用官方提供的**duo-buildroot-sdk**，将**ION_SIZE**修改为0，编译后的镜像可用内存为55M。

### 4.运行
将**runq-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead**、**stories15M.q80**、**tokenizer.bin**传到milkv环境，运行：

`$ ./runq-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead stories15M.q80`
使用第3步编译的镜像运行需要25M swap空间，推理速度约9.5token/s

### 5.优化点
1. int8 Q8_0量化
2. 矩阵向量乘法计算顺序优化


## llama.cpp 版本
该版本在llama.cpp目录下

### 1.编译
`$ make`

### 2.转换
`$ ./convert-llama2c-to-ggml --copy-vocab-from-model llama-2-7b-chat.gguf.q2_K.bin --llama2c-model stories15M.bin --llama2c-output-model stories15M.gguf.bin`

### 3.量化
`$ ./quantize stories15M.gguf.bin stories15M.gguf.q4k q4_0`

### 4.运行
`$ ./main -m stories15M.gguf.q4k -n 100`
使用55M内存版本镜像运行需要2M swap空间，推理速度约5.1token/s

## TTS
由于内存不够，合成速度较慢