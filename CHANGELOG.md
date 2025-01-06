# 更新记录 (CHANGELOG)

## [2025-01-06] - v2.0.0
### 新增

- 在`plot_one_group_bar_figure`和`plot_one_group_violin_figure`函数中，参数`test_method`增加`'external'`选项（即`test_method='external'`）。可在外部计算p值，以列表的形式传入，函数会自动根据p值绘制星号。
- 新增`plot_macaque_brain_figure`函数。原`plot_macaque_brain_figure`更名为`plot_v1_macaque_brain_figure`。
- 新增`plot_macaque_hemi_brain_figure`函数。原`plot_macaque_hemi_brain_figure`更名为`plot_v1_macaque_hemi_brain_figure`。
### 注意

- `plot_v1_macaque_brain_figure`函数将在下个版本中移除
- `plot_v1_macaque_hemi_brain_figure`函数将在下个版本中移除

### 修复

- 无

### 优化
- 优化了猕猴脑图的绘制策略，不再依赖`mne`包，但需要更新本地Python库中的`neuromaps`中的代码。



## [2024-1-1] - v1.0.0

### 新增

- Plot_figure包
