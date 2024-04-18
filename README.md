## Baby LLama2 optimization for Milk-V Duo board

Baby LLama2 is an interesting opensource project, which implements an available LLM chat engine in C code less than 1000 lines. To make such an AI program running on tiny Milk-V duo board (T-Head C906 1GHz with RVV0.7 vector extension, 64MB SDRAM) is a cool thing, which became a challenging problem of first RVSPOC championship (RISCV Software Porting and Optimization Competition) held in 2023, China. We participated in this competition and achieved the optimized story generation speed of 24 tokens/s on offcial Milk-V duo system with 55MB memory configuration. 

## Requirements 

Any Linux system available on Milk-V duo is ok, as long as the free memory is above 25MB (may be queried in terminal by 'free' command, attention to the third column). The available memory of offical Milk-V image is far from enough, which could be improved by recompiling system image with following modification: https://github.com/milkv-duo/duo-buildroot-sdk#faqs

## Optimization 

1. Use int8 quantitized model to reduce memory footprint and improve performance.
2. Use partial on-fly dequantitization to dramatically reduce memory footprint, which greatly improve performance of file I/O cache.
3. Use RVV instrinsic to optimize time-consuming matmul function.
4. Optimize exponential function with approximated fast algorithm

Above optimizations is only applied on int8 code 'runq.c' , while float32 version 'run.c' is unmodified.

## Compilation and Usage

Just execute `make runfast` to obtain the best optimized binary. The default compiler is gcc, while you can use alternative clang compiler by executing `COMPILER=clang make runfast`. The gcc compiler could be downloaded from https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource/1705396260835/Xuantie-900-gcc-linux-5.10.4-musl32-x86_64-V2.8.1-20240115.tar.gz , while clang compiler from https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1701340474733/Xuantie-900-llvm-linux-5.10.4-glibc-x86_64-V1.0.0-beta-20231116.tar.gz . Attention, the Milk-V official gcc compiler 'Xuantie-900 linux-5.10.4 musl gcc Toolchain V2.6.1 B-20220906' could not compile the optimized RVV instrinsic code of matmul function. The reason is still unknown. Any fix is welcome and appreciated. 

Additionally please note that, the gcc compiled binary (runfast version) is about 10% slower than clang compilation, while it is much smaller than clang. The main reason may be that clang support auto vectorization and do more aggressive performance optimization while gcc pays more attention to the balance of speed and size. So we choose gcc as the default compiler. The desicion is up to you according to application scenario.

After compilation, the binary 'runq-fast' could be uploaded to Milk-V board and run with prepared int8 model file stories15M_q80.bin. The command line arguments could be referenced from baby-llama2 README: https://github.com/karpathy/llama2.c . The int8 quantitized model file is not provided by official baby-llama2 repo, while you can download it from prebuilt directory of this repo, or convert it by following steps: 
1. download pretrained weight data stories15M.pt from https://huggingface.co/karpathy/tinyllamas/tree/main
2. convert it to int8 model file by executing `python export.py stories15M_q80.bin --version 2 --checkpoint ./stories15M.pt` . Please note that you should install python3.8 and pytorch first. 

## Prebuilt binary

For the convenience of evaluation, we added prebuilt directory and uploaded the binary executables compiled by gcc and clang, and the int8 model file (compressed for traffic saving, so please unzip it before usage), which could be directly uploaded to Milk-V board and run, enjoy :)
