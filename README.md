# 对stories15M.bin进行int8量化以获得加速推理

### 1.编译

修改Makefile，填写你的编译器路径，查看 .PHONY: milk-v 下的注释获得更多编译相关的问题。

```
[root@milkv-duo]~# make milk-v
```

make之后我们得到了runq二进制文件。

### 2.量化模型

```bash
python3 export.py stories15M_q8.bin --version 2 --checkpoint ./stories15M.pt
```

### 3.运行

将编译好的**runq**二进制文件和int8量化后得到的**stories15M_q8.bin**拖入milk-v duo，运行。

使用int8量化模型运行，需要使用swap。在不使用swap的情况下，只能使用run.c常规的float32推理，无论如何速度都不会超过0.3 token/s，并无实际意义。

```bash
mkswap /dev/mmcblk0p3
```

```bash
./runq stories15M_q8.bin
```

得到结果：

```
Once upon a time, in a quiet little town, there lived a boy named Tim. Tim loved to go outside and play with his friends. One day, Tim and his friends decided to play a game of hide and seek.
As they played, Tim found a big seat under a tree. It was perfect for him to sit on and eat the yummy food from the food his mom made for him. Tim was very happy and said, "Thank you, Mom, for the delicious food!"
Later that day, Tim and his friends had a race to see who could run the fastest. Tim ran as fast as he could, but he fell and hurt his knee. His friends stopped running to help him. Tim felt sad and said, "I'm sorry, I should have asked Mom for help."
The moral of the story is to always ask for help when you need it, and not to complain too much.
achieved tok/s: 1.152501
```

### 4.说明

该测试建立在下载官方镜像情况下，默认total Mem 28.5M，可用空余内存**3.5M**左右，得到测试结果平均约为**1.1 token/s**，使用社区制作的修改过的**55M**内存镜像速度会有显著提升，提升约**5-6**倍。
经过测试，在低可用内存情况下，若以-Ofast编译选项作为基准，在编译选项上做的任何操作对性能的提升可能性微乎其微。
经过测试，使用RVV向量扩展加速计算在低可用内存情况下，对性能的提升微乎其微。
经过测试，若使用**stories260K.bin**模型速度可得到百倍提升，但是输出结果包含乱码，单词拼写错误，无逻辑，所以无实际意义。

CV1800B包含一颗TPU，经研究无法用于llama2格式模型计算加速。

