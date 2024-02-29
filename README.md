## Baby LLama2 在 Milk-V Duo上的优化 （首届RISC-V 软件移植及优化锦标赛赛题之一S2311）

Baby LLama2是个很有趣的开源项目，用几百行的c代码就实现了一个不折不扣的大语言对话生成引擎。 这样一个AI代码要在Milk-V-duo小板子上面流畅跑起来，并且能够实现讲故事机器人的功能确实是个不小的挑战但也是很有意思的事情。 出于此兴趣我用业余时间参加了这次比赛，最终程序在55M内存配置的官方MilkV系统下实现了23~28 token/s的性能。

## 程序运行环境

对系统发行版和库没有特别要求，但是要求系统启动用free命令查看的total memory要在55M以上。官方原始版本total memory只有28M, 可以按照此链接的方法修改重新编译镜像 https://github.com/milkv-duo/duo-buildroot-sdk#faqs

## 程序使用方法

程序使用方法和上游llama.c开源项目完全相同，详见项目链接 https://github.com/karpathy/llama2.c 。 需要注意的是，模型使用的是自己生成的15M 8bit量化版本而不是官方提供的15M float32版本 。

## 提交的内容

本次提交的压缩包rvspoc-work.zip里只包含可执行程序runq-clang和模型文件stories15M_q80.bin ,  源代码等比赛结果公布后再公开。

另外还录制了不同seed下（seed1 ~ seed10，命令行选项-s <n> ）的演示视频， 链接地址： https://pan.baidu.com/s/1JFABnnOAz4cKsk7NfFslCg?pwd=6666 
