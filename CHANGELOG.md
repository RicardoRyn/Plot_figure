# 更新记录 (CHANGELOG)

---

## [2025-03-9] - v0.0.5

### 新增

- 为`plot_macaque_brain_figure`函数增加"BNA"图集，[参考文献](https://doi.org/10.1016/j.scib.2024.03.031)。

### 注意

- 下个版本将更新猕猴脑图的surf.gii文件。

---

## [2025-02-7] - v0.0.4

### 新增

- 为`plot_human_brain_figure`和`plot_macaque_brain_figure`函数增加新功能：
  - 使用`plot=false`参数，可以选择返回`fig`，或者左半球 (`lh_parc`) 和右半球 (`rh_parc`) 的数据。
- 为`plot_human_brain_figure`函数增加"BNA"图集，[参考文献](https://doi.org/10.1093/cercor/bhw157)。

### 优化

- 优化`plot_human_brain_figure`，`plot_macaque_brain_figure`，`plot_human_hemi_brain_figure`和`plot_macaque_hemi_brain_figure`函数的可读性。

---

## [2025-01-13] - v0.0.3

### 优化

- 优化了 Python 包的代码架构，提升了代码的可维护性和扩展性。

### 移除

- `plot_v1_macaque_brain_figure`
- `plot_v1_macaque_hemi_brain_figure`

---

## [2025-01-09] - v0.0.2

### 新增

- 为所有脑图函数增加`rjx_colorbar`。拥有更多可自定义的选项，可绘制复杂的colorbar。

### 修复

- 修复部分情况下，值小于1时，脑图上不显示颜色的bug。

---

## [2025-01-06] - v0.0.1

### 新增

- 在`plot_one_group_bar_figure`和`plot_one_group_violin_figure`函数中，参数`test_method`增加`'external'`选项（即`test_method='external'`）。可在外部计算p值，以列表的形式传入，函数会自动根据p值绘制星号。
- 新增`plot_macaque_brain_figure`函数。原`plot_macaque_brain_figure`更名为`plot_v1_macaque_brain_figure`。
- 新增`plot_macaque_hemi_brain_figure`函数。原`plot_macaque_hemi_brain_figure`更名为`plot_v1_macaque_hemi_brain_figure`。

### 重大更改

- `plot_v1_macaque_brain_figure`函数将在稍后几个版本中移除
- `plot_v1_macaque_hemi_brain_figure`函数将在稍后几个版本中移除

### 优化
- 优化了猕猴脑图的绘制策略，不再依赖`mne`包，但需要更新本地Python库中的`neuromaps`中的代码。

---

## [2024-1-1] - v0.0.0

### 新增
- Plot_figure包
