# 一. plotfig简介

`plotfig`是一个用于认知神经领域科研绘图的python包。

![Plot_figure](https://imgur.com/3CEDdxc.png)

使用示例见`example.ipynb`。

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

# 二. 依赖

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `nibabel`
- `surfplot`
- `mne-connectivity`
- `plotly`

# 三. 安装

通过`pip`安装：

```bash
pip install plotfig
```