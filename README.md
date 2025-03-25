# 写在最前面

## 重要！！！修改原neuromaps库中的部分文件！！！重要

1. 请将`<python_package_path>/neuromaps/datasets/atlases.py`替换为`Plot_figure/temp/atlases.py`
2. 请将`<python_package_path>/neuromaps/datasets/__init__.py`替换为`Plot_figure/temp/__init__.py`
3. 请将`<python_package_path>/neuromaps/datasets/data/osf.json`替换为`Plot_figure/temp/osf.json`


# Plot Figure

**使用示例见`example.ipynb`，示例图见文件夹`example_figures`（更新比较勤）**

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
- `statsmodels`
- `scipy`
- `mne-connectivity`
- `mne`
- `neuromaps` **（需要修改该包中的部分文件）**
- `surfplot`
- `plotly`

安装包均可在终端使用以下命令完成：

```bash
pip install <package_name> -i https://pypi.tuna.tsinghua.edu.cn/simple
```
