import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_connectivity.viz import plot_connectivity_circle


def plot_symmetric_circle_figure(connectome, labels=None, node_colors=None, vmin=None, vmax=None, figsize=(10, 10), labes_fontsize=15, face_color='w', nodeedge_color='w', text_color='k', cmap='bwr', linewidth=1, title_name='', title_fontsize=20, colorbar=False, colorbar_size=0.2, colorbar_fontsize=10, colorbar_pos=(0, 0), manual_colorbar=False, manual_colorbar_pos=[1, 0.4, 0.01, 0.2], manual_cmap='bwr', manual_colorbar_name='', manual_colorbar_label_fontsize=10, manual_colorbar_fontsize=10, manual_colorbar_rotation=-90, manual_colorbar_pad=20, manual_colorbar_draw_border=True, manual_colorbar_tickline=False, manual_colorbar_nticks=False):
    # 设置默认值
    if vmax is None:
        vmax = np.max((np.max(connectome), -np.min(connectome)))
    if vmin is None:
        vmin = np.min((np.min(connectome), -np.max(connectome)))
    count = connectome.shape[0]
    count_half = int(count/2)
    if labels is None:
        labels = [str(i) for i in range(count_half)]
    if node_colors is None:
        node_colors = ['#ff8f8f'] * count_half
    labels = labels + [i+' ' for i in labels[::-1]]
    node_colors = node_colors + [i for i  in node_colors[::-1]]
    node_angles = mne.viz.circular_layout(labels, labels, group_boundaries=[0, len(labels) / 2])
    # 常规矩阵需要做对称转换
    data_upper_left = connectome[0:count_half, 0:count_half]
    data_down_right = connectome[count_half:count, count_half:count]
    data_down_left = connectome[count_half:count, 0:count_half]
    data_upper_right = connectome[0:count_half, count_half:count]
    data_down_right = data_down_right[::-1][:,::-1]
    data_down_left = data_down_left[::-1]
    data_upper_right = data_upper_right[:,::-1]
    connectome_upper = np.concatenate((data_upper_left, data_upper_right), axis=1)
    connectome_lower = np.concatenate((data_down_left, data_down_right), axis=1)
    connectome = np.concatenate((connectome_upper, connectome_lower), axis=0)
    # 画图
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plot_connectivity_circle(connectome, labels, node_angles=node_angles, node_colors=node_colors,fontsize_names=labes_fontsize, facecolor=face_color, node_edgecolor=nodeedge_color, textcolor=text_color, colormap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, title=title_name, fontsize_title=title_fontsize, colorbar=colorbar, colorbar_size=colorbar_size, colorbar_pos=colorbar_pos, fontsize_colorbar=colorbar_fontsize, fig=fig, ax=ax, interactive=False, show=False)
    # 如有需要，禁用自动colorbar，手动生成colorbar
    if manual_colorbar:
        # 手动创建colorbar，拥有更多的设置
        cax = fig.add_axes(manual_colorbar_pos)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=manual_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.outline.set_visible(manual_colorbar_draw_border)
        cbar.ax.set_ylabel(manual_colorbar_name, fontsize=manual_colorbar_label_fontsize, rotation=manual_colorbar_rotation, labelpad=manual_colorbar_pad)
        if not manual_colorbar_tickline:
            cbar.ax.tick_params(length=0)  # 不显示竖线
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=manual_colorbar_fontsize)
        if manual_colorbar_nticks:
            ticks = np.linspace(vmin, vmax, manual_colorbar_nticks)
            cbar.set_ticks(ticks)
    return fig

def plot_asymmetric_circle_figure(connectome, labels=None, node_colors=None, vmin=None, vmax=None, figsize=(10, 10), labes_fontsize=15, face_color='w', nodeedge_color='w', text_color='k', cmap='bwr', linewidth=1, title_name='', title_fontsize=20, colorbar=False, colorbar_size=0.2, colorbar_fontsize=10, colorbar_pos=(0, 0), manual_colorbar=False, manual_colorbar_pos=[1, 0.4, 0.01, 0.2], manual_cmap='bwr', manual_colorbar_name='', manual_colorbar_label_fontsize=10, manual_colorbar_fontsize=10, manual_colorbar_rotation=-90, manual_colorbar_pad=20, manual_colorbar_draw_border=True, manual_colorbar_tickline=False, manual_colorbar_nticks=False):
    # 设置默认值
    if vmax is None:
        vmax = np.max((np.max(connectome), -np.min(connectome)))
    if vmin is None:
        vmin = np.min((np.min(connectome), -np.max(connectome)))
    count = connectome.shape[0]
    if labels is None:
        labels = [str(i) for i in range(count)]
    if node_colors is None:
        node_colors = ['#ff8f8f'] * count
    node_angles = mne.viz.circular_layout(labels, labels)
    # 画图
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plot_connectivity_circle(connectome, labels, node_angles=node_angles, node_colors=node_colors,fontsize_names=labes_fontsize, facecolor=face_color, node_edgecolor=nodeedge_color, textcolor=text_color, colormap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, title=title_name, fontsize_title=title_fontsize, colorbar=colorbar, colorbar_size=colorbar_size, colorbar_pos=colorbar_pos, fontsize_colorbar=colorbar_fontsize, fig=fig, ax=ax, interactive=False, show=False)
    # 如有需要，禁用自动colorbar，手动生成colorbar
    if manual_colorbar:
        # 手动创建colorbar，拥有更多的设置
        cax = fig.add_axes(manual_colorbar_pos)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=manual_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.outline.set_visible(manual_colorbar_draw_border)
        cbar.ax.set_ylabel(manual_colorbar_name, fontsize=manual_colorbar_label_fontsize, rotation=manual_colorbar_rotation, labelpad=manual_colorbar_pad)
        if not manual_colorbar_tickline:
            cbar.ax.tick_params(length=0)  # 不显示竖线
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=manual_colorbar_fontsize)
        if manual_colorbar_nticks:
            ticks = np.linspace(vmin, vmax, manual_colorbar_nticks)
            cbar.set_ticks(ticks)
    return fig
