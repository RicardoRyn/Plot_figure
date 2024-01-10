<hr><center><h1>Plot_figure</h1></center><hr>

这是一个用来画统计图的python脚本。目前已经可以绘制以下图片：

1. 单组bar图
2. 多组bar图
3. 矩阵图
4. 单组小提琴图
5. 多组小提琴图
6. 点线相关图

# 1. 单组bar图

使用例：

```python
import numpy as np
import matplotlib.pyplot as plt
from plot_figure import *

# 原始数据
Human = np.random.normal(1000,100,100)
Random = np.random.normal(1500,100,100)
Macaque = np.random.normal(2000,100,100)

# 导入数据
data = [Human, Random, Macaque]
labels_name = ['Human', 'Random', 'Macaque']
colors = ['#c44e52','#bcbbc0', '#1a1a1a']

# 设置figure
fig = plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# 设置axes
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# 调用函数
plot_one_group_bar_figure(data, test_method='ttest_rel', ax=ax1, colors=colors, y_max_tick_to_one=True, y_max_tick_to_value=1000)
plot_one_group_bar_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax2, colors=colors, math_text=False, one_decimal_place=True, x_tick_rotation=30, x_label_ha='right')
plot_one_group_bar_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax3, colors=colors, percentage=True, math_text=False)
plot_one_group_bar_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax4, colors=colors, ax_min_is_0=True)

# 保存图片
fig.savefig("D:\\Desktop\\rm_rjx.svg", dpi=250, bbox_inches='tight')
```

# 2. 多组bar图

使用例：

```python
# 原始数据
Human1 = np.random.normal(1000, 10, 100)
Human2 = np.random.normal(990, 10, 100)
Human3 = np.random.normal(980, 10, 100)
Macaque1 = np.random.normal(970, 10, 90)
Macaque2 = np.random.normal(990, 10, 90)
Macaque3 = np.random.normal(950, 10, 90)
Dog1 = np.random.normal(970, 10, 90)
Dog2 = np.random.normal(970, 10, 90)
Dog3 = np.random.normal(970, 10, 90)
Human = np.concatenate((Human1.reshape(-1, 1), Human2.reshape(-1, 1), Human3.reshape(-1, 1)), axis=1)  # 100，3
Macaque = np.concatenate((Macaque1.reshape(-1, 1), Macaque2.reshape(-1, 1), Macaque3.reshape(-1, 1)), axis=1)  # 90，3
Dog = np.concatenate((Dog1.reshape(-1, 1), Dog2.reshape(-1, 1), Dog3.reshape(-1, 1)), axis=1)  # 90，3
# 导入数据
data = [Human, Macaque, Dog]
legend_name = ['Human', 'Macaque', 'Dog']
labels_name = ['MST', 'MT', 'FST']
colors = ['#c44e52', '#1a1a1a', '#bbbbc0']

# 设置figure
fig = plt.figure(figsize=(5,5))
# 设置axes
ax = fig.add_subplot(111)
# 调用函数
plot_multi_group_bar_figure(data, test_method='ttest_ind', legend_name=legend_name, labels_name=labels_name, colors=colors, legend_location='lower left')
# 保存图片
fig.savefig(r"D:\Desktop\rm_rjx.svg",dpi=250, bbox_inches='tight')
```

# 3. 矩阵图

使用例：

```python
# 数据
data = np.random.random((88,88))
labels = ['area_32','area_25','area_24a/b','area_24c','area_24a/b_prime','area_24c_prime','area_10','area_14','area_11','area_13','area_12m/o','Iam/Iapm','lat_Ia','OLF','G','PrCO','area_8A','area_8B','area_9','area_46d','area_46v/f','area_12r/l','area_45','area_44','M1','PMd','PMv','preSMA','SMA','area_3a/b','areas_1-2','SII','V6','V6A','area_5d','PEa','MIP','fundus_IPS','AIP','LIP','LOP','MST','area_7a/b','area_7op','area_7m','area_31','area_23','area_v23','area_29','area_30','TF/TFO','TH','caudal_ERh','mid_ERh','rostral_ERh','area_35','area_36','TGa','TGd','TGg','TEO','post_TE','ant_TE','TE_in_STSv','ant_STSf','FST','TPO','TAa','STGr','Tpt','parabelt','CL/ML','AL/RTL','CM','RM/RTM','RTp','R/RT','AI','Pi','Ins','Ri','MT','V4d','V4v','V3d/V3A','V3v','V2','V1']

# figure的参数额外设置
fig = plt.figure(figsize=(20,10))
# axes的参数额外设置
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# 调用函数
plot_matrix_figure(data, row_labels_name=labels, col_labels_name=labels, ax=ax1, cmap='Reds', cbarlabel_name='BBB', title_name='AAA', vmin=0, vmax=0.1)
plot_matrix_figure(data, ax=ax2)
# 保存图片
fig.savefig("D:\\Desktop\\rm_rjx.svg", dpi=250, bbox_inches='tight')
```

# 4. 单组小提琴图

使用例：

```python
# 原始数据
Human = np.random.normal(1000,100,100)
Random = np.random.normal(1500,100,100)
Macaque = np.random.normal(2000,100,100)
# 导入数据
data=[Human, Random, Macaque]
labels_name = ['Human', 'Random', 'Macaque']
colors=['#c44e52','#bcbbc0', '#1a1a1a']

# 设置figure
fig = plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# 设置axes
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# 调用函数
plot_one_group_violin_figure(data, test_method='ttest_rel', ax=ax1, colors=colors, y_max_tick_to_one=True, y_max_tick_to_value=1000)
plot_one_group_violin_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax2, colors=colors, math_text=False, one_decimal_place=True, x_tick_rotation=30, x_label_ha='right')
plot_one_group_violin_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax3, colors=colors, percentage=True, math_text=False)
plot_one_group_violin_figure(data, test_method='ttest_rel', labels_name=labels_name, ax=ax4, colors=colors, ax_min_is_0=True)
# 保存图片
fig.savefig("D:\\Desktop\\rm_rjx.svg", dpi=250, bbox_inches='tight')
```

