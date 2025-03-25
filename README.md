# Plot Figure

使用示例见`example.ipynb`

![Plot_figure](https://imgur.com/3CEDdxc.png)

# 一. 简介

`Plot_figure`是一个用于认知神经领域科研绘图的python包。

主要包括图的种类：
1. 单组bar图
1. 单组小提琴图
1. 矩阵图
1. 点线相关图
1. 脑图
   1. 人类Glasser脑区图
   1. 人类BNA脑区图
   1. 猕猴CHARM5脑区图
   1. 猕猴CHARM6脑区图
   1. 猕猴BNA脑区图
1. 圈状图（circos图）
   1. 对称circos图
   1. 不对称circos图
1. 大脑连接图
   1. 猕猴大脑连接图

> 基本知识，一张图中的基本元素的名字：
>
> <img src="https://matplotlib.org/stable/_images/anatomy.png" alt="../../_images/anatomy.png" style="zoom: 33%;" />

# 二. python依赖

## 1. python包的安装

使用前需安装：

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `nibabel`
- `surfplot`
- `mne-connectivity`
- `plotly`

安装包均可在终端使用以下命令完成：

```bash
pip install <package_name> -i https://pypi.tuna.tsinghua.edu.cn/simple
```
