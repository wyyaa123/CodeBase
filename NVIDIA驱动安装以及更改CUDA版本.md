# Ubuntu 18.04 🚀NVIDIA驱动安装以及更改CUDA版本

## 一、为什么写这篇文章

-   **背景**：以前在跑卷积神经网络时，并没有深入的去了解这方面的基础知识，只是在装完显卡驱动后直接使用torch库跑代码；今天正好在跑模型有一点时间和想法，正是个写经验贴的好时节啊！😎

```python
import torch
torch.cuda.is_available()

>>> True
#这个时候天真的以为输出True了，给模型喂数据也没有出现kernel错误就🆗了
```

-   **出发点**：前几天跑一个开源超分框架[**BasicSR**](https://github.com/XPixelGroup/BasicSR)，疯狂报错。首先仔细对比了第三方库的版本发现torch版本有问题，换完torch版本后又出现了CUDA的kernel问题，这才了解到cuda，torch，torchvision的版本必须要兼容。为此查了很多资料，这些资料又杂又乱总不能以后再要用的时候又重新花费时间去找资料吧。不如自己整一个！✊

<center><b>好啦正题开始</b></center>

## 二、显卡驱动和CUDA的关系

显卡驱动和 CUDA 之间的关系是：CUDA 运行需要显卡驱动程序的支持。显卡驱动提供了显卡的底层接口和支持库，以及与操作系统和计算机应用程序的通信机制，使得 CUDA 能够通过显卡驱动与显卡进行通信并实现计算任务的加速。因此，在使用 CUDA 之前，需要先安装与显卡相应的显卡驱动程序。

​																								——摘自 [显卡、显卡驱动、cuda的关系](https://blog.csdn.net/Maggie_JK/article/details/132825301)

## 三、前期准备

我平时的开发环境都是在Windows上基于SSH钥匙远程连接实验室的服务器，这里不得不提地表最强的开发神器VScode😍，以及两款远程软件MobaXterm和VNC。

要做到VScode远程开发需要Windows上安装SSH，我用的是Windows11自带了SSH，Win10的同学可以去搜搜VScode+SSH这两个关键词啊。