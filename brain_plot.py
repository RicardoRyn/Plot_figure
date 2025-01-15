import os
import os.path as op
import math

import numpy as np
import pandas as pd
import nibabel as nib

from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable


from neuromaps.datasets import fetch_fslr
from neuromaps.datasets import fetch_rjx_hcpmacaque
from surfplot import Plot


def plot_human_brain_figure(data, surf='veryinflated', atlas='glasser', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name='', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False, title_name='', title_fontsize=15, title_y=0.9, rjx_colorbar=False, rjx_colorbar_direction='vertical', horizontal_center=True, rjx_colorbar_outline=False, rjx_colorbar_label_name='', rjx_colorbar_tick_fontsize=10, rjx_colorbar_label_fontsize=10, rjx_colorbar_tick_rotation=0, rjx_colorbar_tick_length=0, rjx_colorbar_nticks=2):
    '''
    surf的种类有：veryinflated, inflated, midthickness, sphere
    '''
    # 设置必要文件路径
    current_dir = os.path.dirname(__file__)
    neuromaps_data_dir = op.join(current_dir, 'neuromaps-data')
    if atlas == 'glasser':
        lh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.L.Glasser.32k_fs_LR.label.gii')
        rh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.R.Glasser.32k_fs_LR.label.gii')
    elif atlas == 'bna':
        lh_BNA_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_BNA_atlas', 'fsaverage.L.BNA.32k_fs_LR.label.gii')
        rh_BNA_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_BNA_atlas', 'fsaverage.R.BNA.32k_fs_LR.label.gii')
    # 获取文件Underlay
    surfaces = fetch_fslr(data_dir=neuromaps_data_dir)
    lh, rh = surfaces[surf]
    p = Plot(lh, rh)
    # 将原始数据拆分成左右脑数据
    lh_data, rh_data = {}, {}
    for hemi_data in data:
        if 'lh_' in hemi_data:
            lh_data[hemi_data] = data[hemi_data]
        else:
            rh_data[hemi_data] = data[hemi_data]
    # 加载图集分区数据
    if atlas == 'glasser':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'human_glasser.csv'))
    elif atlas == 'bna':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'human_bna.csv'))

    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]

    if atlas == 'glasser':
        lh_parc = nib.load(lh_Glasser_atlas_dir).darrays[0].data
        roi_vertics = {roi:[] for roi in lh_roi_list}
        for index, roi_index in enumerate(lh_parc):
            if roi_index-181 >= 0:
                roi_vertics[lh_roi_list[roi_index-181]].append(index)
        lh_parc = np.zeros_like(lh_parc).astype('float32')
        for roi_data in lh_data:
            lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    elif atlas == 'bna':
        lh_parc = nib.load(lh_BNA_atlas_dir).darrays[0].data
        roi_vertics = {roi:[] for roi in lh_roi_list}
        for index, roi_index in enumerate(lh_parc):
            if roi_index-1 >= 0:
                roi_vertics[lh_roi_list[roi_index-1]].append(index)
        lh_parc = np.zeros_like(lh_parc).astype('float32')
        for roi_data in lh_data:
            lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]



    if atlas == 'glasser':
        rh_parc = nib.load(rh_Glasser_atlas_dir).darrays[0].data
        roi_vertics = {roi:[] for roi in rh_roi_list}
        for index, roi_index in enumerate(rh_parc):
            if roi_index-1 >= 0:
                roi_vertics[rh_roi_list[roi_index-1]].append(index)
        rh_parc = np.zeros_like(rh_parc).astype('float32')
        for roi_data in rh_data:
            rh_parc[roi_vertics[roi_data]] = rh_data[roi_data]
    elif atlas == 'bna':
        rh_parc = nib.load(rh_BNA_atlas_dir).darrays[0].data
        roi_vertics = {roi:[] for roi in rh_roi_list}
        for index, roi_index in enumerate(rh_parc):
            if roi_index-106 >= 0:
                roi_vertics[rh_roi_list[roi_index-106]].append(index)
        rh_parc = np.zeros_like(rh_parc).astype('float32')
        for roi_data in rh_data:
            rh_parc[roi_vertics[roi_data]] = rh_data[roi_data]
        

    # 画图元素参数设置
    if vmin is None:
        vmin = min(data.values())
    if vmax is None:
        vmax = max(data.values())
    if vmin > vmax:
        print('vmin必须小于等于vmax')
        return
    if vmin == vmax:
        vmin = min(0, vmin)
        vmax = max(0, vmax)
    # colorbar参数设置
    colorbar_kws = {'location': colorbar_location, 'label_direction': colorbar_label_rotation, 'decimals': colorbar_decimals, 'fontsize': colorbar_fontsize, 'n_ticks': colorbar_nticks, 'shrink': colorbar_shrink, 'aspect': colorbar_aspect, 'draw_border': colorbar_draw_border}
    p.add_layer({'left': lh_parc, 'right': rh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)
    fig = p.build(cbar_kws=colorbar_kws)
    fig.suptitle(title_name, fontsize=title_fontsize, y=title_y)
    ############################################### rjx_colorbar ###############################################
    sm = ScalarMappable(cmap=cmap)
    sm.set_array((vmin, vmax))  # 设置值范围
    if rjx_colorbar:
        if rjx_colorbar_direction == 'vertical':
            formatter = ScalarFormatter(useMathText=True)  # 科学计数法相关
            formatter.set_powerlimits((-3, 3))  # <=-1也就是小于等于0.1，>=2，也就是大于等于100，会写成科学计数法
            cax = fig.add_axes([1, 0.425, 0.01, 0.15])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', cmap=cmap)  # "vertical", "horizontal"
            cbar.outline.set_visible(rjx_colorbar_outline)
            cbar.ax.set_ylabel(rjx_colorbar_label_name, fontsize=rjx_colorbar_label_fontsize)
            cbar.ax.yaxis.set_label_position("left")  # 原本设置y轴label默认在右边，现在换到左边
            cbar.ax.tick_params(axis='y', which='major', labelsize=rjx_colorbar_tick_fontsize, rotation=rjx_colorbar_tick_rotation, length=rjx_colorbar_tick_length)
            cbar.set_ticks(np.linspace(vmin, vmax, rjx_colorbar_nticks))
            if vmax < 0.001 or vmax > 1000:  # y轴设置科学计数法
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.get_offset_text().set_visible(False)  # 隐藏默认的偏移文本
                exponent = math.floor(math.log10(vmax))
                # 手动添加文本
                cbar.ax.text(1.05, 1.15, rf"$\times 10^{{{exponent}}}$", transform=cbar.ax.transAxes, fontsize=rjx_colorbar_tick_fontsize, verticalalignment='bottom', horizontalalignment='left')
        elif rjx_colorbar_direction == 'horizontal':
            if horizontal_center:
                cax = fig.add_axes([0.44, 0.5, 0.15, 0.01])
            else:
                cax = fig.add_axes([0.44, 0.05, 0.15, 0.01])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', cmap=cmap)
            cbar.outline.set_visible(rjx_colorbar_outline)
            cbar.ax.set_title(rjx_colorbar_label_name, fontsize=rjx_colorbar_label_fontsize)
            cbar.ax.tick_params(axis='x', which='major', labelsize=rjx_colorbar_tick_fontsize, rotation=rjx_colorbar_tick_rotation, length=rjx_colorbar_tick_length)
            if vmax < 0.001 or vmax > 1000:  # y轴设置科学计数法
                cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.set_ticks([vmin, vmax])
    return fig

def plot_human_hemi_brain_figure(data, hemi='lh', surf='veryinflated', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name= '', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False, title_name='', title_fontsize=15, title_y=0.9):
    '''
    surf的种类有：veryinflated, inflated, midthickness, sphere
    '''
    # 设置必要文件路径
    current_dir = os.path.dirname(__file__)
    neuromaps_data_dir = op.join(current_dir, 'neuromaps-data')
    lh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.L.Glasser.32k_fs_LR.label.gii')
    rh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.R.Glasser.32k_fs_LR.label.gii')
    # 获取文件Underlay
    surfaces = fetch_fslr(data_dir=neuromaps_data_dir)
    lh, rh = surfaces[surf]
    if hemi == 'lh':
        p = Plot(lh, size=(800, 400), zoom=1.2)
    else:
        p = Plot(rh, size=(800, 400), zoom=1.2)
    # 将原始数据拆分成左右脑数据
    lh_data, rh_data = {}, {}
    for hemi_data in data:
        if 'lh_' in hemi_data:
            lh_data[hemi_data] = data[hemi_data]
        else:
            rh_data[hemi_data] = data[hemi_data]
    # 加载Glasser分区数据
    df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'human_glasser.csv'))
    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    lh_parc = nib.load(lh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in lh_roi_list}
    for index, roi_index in enumerate(lh_parc):
        if roi_index-181 >= 0:
            roi_vertics[lh_roi_list[roi_index-181]].append(index)
    lh_parc = np.zeros_like(lh_parc).astype('float32')
    for roi_data in lh_data:
        lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    rh_parc = nib.load(rh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in rh_roi_list}
    for index, roi_index in enumerate(rh_parc):
        if roi_index-1 >= 0:
            roi_vertics[rh_roi_list[roi_index-1]].append(index)
    rh_parc = np.zeros_like(rh_parc).astype('float32')
    for roi_data in rh_data:
        rh_parc[roi_vertics[roi_data]] = rh_data[roi_data]
    # 画图元素参数设置
    if vmin is None:
        vmin = min(data.values())
    if vmax is None:
        vmax = max(data.values())
    if vmin > vmax:
        print('vmin必须小于等于vmax')
        return
    if vmin == vmax:
        vmin = min(0, vmin)
        vmax = max(0, vmax)
    # colorbar参数设置
    colorbar_kws = {'location': colorbar_location, 'label_direction': colorbar_label_rotation, 'decimals': colorbar_decimals, 'fontsize': colorbar_fontsize, 'n_ticks': colorbar_nticks, 'shrink': colorbar_shrink, 'aspect': colorbar_aspect, 'draw_border': colorbar_draw_border}
    if hemi == 'lh':
        p.add_layer({'left': lh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)
    else:
        p.add_layer({'left': rh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)  # 很怪，但是这里就是写“{'left': rh_parc}”
    fig = p.build(cbar_kws=colorbar_kws)
    fig.suptitle(title_name, fontsize=title_fontsize, y=title_y)
    return fig

def plot_macaque_brain_figure(data, surf='veryinflated', atlas='charm5', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name='', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False, title_name='', title_fontsize=15, title_y=0.9, rjx_colorbar=False, rjx_colorbar_direction='vertical', horizontal_center=True, rjx_colorbar_outline=False, rjx_colorbar_label_name='', rjx_colorbar_tick_fontsize=10, rjx_colorbar_label_fontsize=10, rjx_colorbar_tick_rotation=0, rjx_colorbar_tick_length=0, rjx_colorbar_nticks=2):
    '''
    surf的种类有：veryinflated, inflated, midthickness, sphere, pial
    '''
    # 设置必要文件路径
    current_dir = os.path.dirname(__file__)
    neuromaps_data_dir = op.join(current_dir, 'neuromaps-data')
    lh_CHARM5_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM5_atlas', 'L.charm5.label.gii')
    rh_CHARM5_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM5_atlas', 'R.charm5.label.gii')
    lh_CHARM6_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM6_atlas', 'L.charm6.label.gii')
    rh_CHARM6_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM6_atlas', 'R.charm6.label.gii')
    # 获取文件Underlay
    surfaces = fetch_rjx_hcpmacaque(data_dir=neuromaps_data_dir)
    lh, rh = surfaces[surf]
    p = Plot(lh, rh)
    # 将原始数据拆分成左右脑数据
    lh_data, rh_data = {}, {}
    for hemi_data in data:
        if 'lh_' in hemi_data:
            lh_data[hemi_data] = data[hemi_data]
        else:
            rh_data[hemi_data] = data[hemi_data]
    # 加载分区数据
    if atlas == 'charm5':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'macaque_charm5.csv'))
    elif atlas == 'charm6':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'macaque_charm6.csv'))
    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    if atlas == 'charm5':
        rh_parc = nib.load(rh_CHARM5_atlas_dir).darrays[0].data
    elif atlas == 'charm6':
        rh_parc = nib.load(rh_CHARM6_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in rh_roi_list}
    for index, roi_index in enumerate(rh_parc):
        if roi_index-len(lh_roi_list)-1 >= 0:
            roi_vertics[rh_roi_list[roi_index-len(lh_roi_list)-1]].append(index)
    rh_parc = np.zeros_like(rh_parc).astype('float32')
    for roi_data in rh_data:
        rh_parc[roi_vertics[roi_data]] = rh_data[roi_data]
    if atlas == 'charm5':
        lh_parc = nib.load(lh_CHARM5_atlas_dir).darrays[0].data
    elif atlas == 'charm6':
        lh_parc = nib.load(lh_CHARM6_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in lh_roi_list}
    for index, roi_index in enumerate(lh_parc):
        if roi_index-1 >= 0:
            roi_vertics[lh_roi_list[roi_index-1]].append(index)
    lh_parc = np.zeros_like(lh_parc).astype('float32')
    for roi_data in lh_data:
        lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    # 画图元素参数设置
    if vmin is None:
        vmin = min(data.values())
    if vmax is None:
        vmax = max(data.values())
    if vmin > vmax:
        print('vmin必须小于等于vmax')
        return
    if vmin == vmax:
        vmin = min(0, vmin)
        vmax = max(0, vmax)
    # colorbar参数设置
    colorbar_kws = {'location': colorbar_location, 'label_direction': colorbar_label_rotation, 'decimals': colorbar_decimals, 'fontsize': colorbar_fontsize, 'n_ticks': colorbar_nticks, 'shrink': colorbar_shrink, 'aspect': colorbar_aspect, 'draw_border': colorbar_draw_border}
    p.add_layer({'left': lh_parc, 'right': rh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)
    fig = p.build(cbar_kws=colorbar_kws)
    fig.suptitle(title_name, fontsize=title_fontsize, y=title_y)
    ############################################### rjx_colorbar ###############################################
    sm = ScalarMappable(cmap=cmap)
    sm.set_array((vmin, vmax))  # 设置值范围
    if rjx_colorbar:
        formatter = ScalarFormatter(useMathText=True)  # 科学计数法相关
        formatter.set_powerlimits((-3, 3))  # <=-1也就是小于等于0.1，>=2，也就是大于等于100，会写成科学计数法
        if rjx_colorbar_direction == 'vertical':
            cax = fig.add_axes([1, 0.425, 0.01, 0.15])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', cmap=cmap)  # "vertical", "horizontal"
            cbar.outline.set_visible(rjx_colorbar_outline)
            cbar.ax.set_ylabel(rjx_colorbar_label_name, fontsize=rjx_colorbar_label_fontsize)
            cbar.ax.yaxis.set_label_position("left")  # 原本设置y轴label默认在右边，现在换到左边
            cbar.ax.tick_params(axis='y', which='major', labelsize=rjx_colorbar_tick_fontsize, rotation=rjx_colorbar_tick_rotation, length=rjx_colorbar_tick_length)
            cbar.set_ticks(np.linspace(vmin, vmax, rjx_colorbar_nticks))
            if vmax < 0.001 or vmax > 1000:  # y轴设置科学计数法
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.get_offset_text().set_visible(False)  # 隐藏默认的偏移文本
                exponent = math.floor(math.log10(vmax))
                # 手动添加文本
                cbar.ax.text(1.05, 1.15, rf"$\times 10^{{{exponent}}}$", transform=cbar.ax.transAxes, fontsize=rjx_colorbar_tick_fontsize, verticalalignment='bottom', horizontalalignment='left')
        elif rjx_colorbar_direction == 'horizontal':
            if horizontal_center:
                cax = fig.add_axes([0.44, 0.5, 0.15, 0.01])
            else:
                cax = fig.add_axes([0.44, 0.05, 0.15, 0.01])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', cmap=cmap)
            cbar.outline.set_visible(rjx_colorbar_outline)
            cbar.ax.set_title(rjx_colorbar_label_name, fontsize=rjx_colorbar_label_fontsize)
            cbar.ax.tick_params(axis='x', which='major', labelsize=rjx_colorbar_tick_fontsize, rotation=rjx_colorbar_tick_rotation, length=rjx_colorbar_tick_length)
            if vmax < 0.001 or vmax > 1000:  # y轴设置科学计数法
                cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.set_ticks([vmin, vmax])
    return fig

def plot_macaque_hemi_brain_figure(data, hemi='lh', surf='veryinflated', atlas='charm5', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name= '', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False, title_name='', title_fontsize=15, title_y=0.9):
    '''
    surf的种类有：veryinflated, inflated, midthickness, sphere
    '''
    # 设置必要文件路径
    current_dir = os.path.dirname(__file__)
    neuromaps_data_dir = op.join(current_dir, 'neuromaps-data')
    lh_CHARM5_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM5_atlas', 'L.charm5.label.gii')
    rh_CHARM5_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM5_atlas', 'R.charm5.label.gii')
    lh_CHARM6_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM6_atlas', 'L.charm6.label.gii')
    rh_CHARM6_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_CHARM6_atlas', 'R.charm6.label.gii')
    # 获取文件Underlay
    surfaces = fetch_rjx_hcpmacaque(data_dir=neuromaps_data_dir)
    lh, rh = surfaces[surf]
    if hemi == 'lh':
        p = Plot(lh, size=(800, 400), zoom=1.2)
    else:
        p = Plot(rh, size=(800, 400), zoom=1.2)
    # 将原始数据拆分成左右脑数据
    lh_data, rh_data = {}, {}
    for hemi_data in data:
        if 'lh_' in hemi_data:
            lh_data[hemi_data] = data[hemi_data]
        else:
            rh_data[hemi_data] = data[hemi_data]
    # 加载分区数据
    if atlas == 'charm5':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'macaque_charm5.csv'))
    elif atlas == 'charm6':
        df = pd.read_csv(op.join(current_dir, 'atlas_tables', 'macaque_charm6.csv'))
    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    if atlas == 'charm5':
        lh_parc = nib.load(lh_CHARM5_atlas_dir).darrays[0].data
    elif atlas == 'charm6':
        lh_parc = nib.load(lh_CHARM6_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in lh_roi_list}
    for index, roi_index in enumerate(lh_parc):
        if roi_index-1 >= 0:
            roi_vertics[lh_roi_list[roi_index-1]].append(index)
    lh_parc = np.zeros_like(lh_parc).astype('float32')
    for roi_data in lh_data:
        lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    if atlas == 'charm5':
        rh_parc = nib.load(rh_CHARM5_atlas_dir).darrays[0].data
    elif atlas == 'charm6':
        rh_parc = nib.load(rh_CHARM6_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in rh_roi_list}
    for index, roi_index in enumerate(rh_parc):
        if roi_index-len(lh_roi_list)-1 >= 0:
            roi_vertics[rh_roi_list[roi_index-len(lh_roi_list)-1]].append(index)
    rh_parc = np.zeros_like(rh_parc).astype('float32')
    for roi_data in rh_data:
        rh_parc[roi_vertics[roi_data]] = rh_data[roi_data]
    # 画图元素参数设置
    if vmin is None:
        vmin = min(data.values())
    if vmax is None:
        vmax = max(data.values())
    if vmin > vmax:
        print('vmin必须小于等于vmax')
        return
    if vmin == vmax:
        vmin = min(0, vmin)
        vmax = max(0, vmax)
    # colorbar参数设置
    colorbar_kws = {'location': colorbar_location, 'label_direction': colorbar_label_rotation, 'decimals': colorbar_decimals, 'fontsize': colorbar_fontsize, 'n_ticks': colorbar_nticks, 'shrink': colorbar_shrink, 'aspect': colorbar_aspect, 'draw_border': colorbar_draw_border}
    if hemi == 'lh':
        p.add_layer({'left': lh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)
    else:
        p.add_layer({'left': rh_parc}, cbar=colorbar, cmap=cmap, color_range=(vmin, vmax), cbar_label=colorbar_label_name)  # 很怪，但是这里就是写“{'left': rh_parc}”
    fig = p.build(cbar_kws=colorbar_kws)
    fig.suptitle(title_name, fontsize=title_fontsize, y=title_y)
    return fig

