import os
import os.path as op
from itertools import chain

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
import statsmodels.api as sm

import mne
from neuromaps.datasets import fetch_fslr
from surfplot import Plot



def plot_one_group_bar_figure(data, ax=None, labels_name=None, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', width=0.5, colors=None, title_name='', title_fontsize=10, title_pad=20, x_label_name='', x_label_fontsize=10, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, y_lim_range=None, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, statistic=False, test_method='ttest_ind', asterisk_fontsize=10, multicorrect=False, **kwargs):
    # # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    if labels_name is None:
        labels_name = [str(i) for i in range(len(data))]
    if colors is None:
        colors = ['gray'] * len(data)
    ##################################################################################################
    bar_mean_list, bar_SD_list, bar_SE_list = [], [], []
    for i in range(0,len(labels_name)):
        np.random.seed(1998)
        globals()['data'+str(i)] = data[i]
        globals()['x'+str(i)+'_dots'] = np.random.normal(0,0.1,len(eval('data'+str(i)))) + i + 0
        globals()['bar'+str(i)+'_mean'] = np.mean(eval('data'+str(i)))
        globals()['bar'+str(i)+'_len'] = len(eval('data'+str(i)))
        globals()['bar'+str(i)+'_SD'] = np.std(eval('data'+str(i)))
        globals()['bar'+str(i)+'_SE'] = eval('bar'+str(i)+'_SD') / eval('bar'+str(i)+'_len') ** 0.5
        bar_mean_list.append(eval('bar'+str(i)+'_mean'))
        bar_SD_list.append(eval('bar'+str(i)+'_SD'))
        bar_SE_list.append(eval('bar'+str(i)+'_SE'))
    x = np.arange(len(labels_name))
    ax.bar(x,bar_mean_list,width=width,color=colors,alpha=1, edgecolor='k')
    ax.errorbar(x, bar_mean_list, bar_SE_list, fmt='none', linewidth=1, capsize=3, color='black')
    for i in range(0,len(x)):
        ax.scatter(eval('x'+str(i)+'_dots'), eval('data'+str(i)), c=colors[i], s=35, edgecolors='white', linewidths=1, alpha=0.5)
    ############################################### ax ###############################################
    ax.spines[['top', 'right']].set_visible(False)  # 去掉上边和右边的spine
    ############################################## title #############################################
    ax.set_title(title_name, fontsize=title_fontsize, pad=title_pad)
    ################################################ x ###############################################
    ax.set_xlabel(x_label_name, fontsize=x_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_name, ha=x_label_ha, rotation_mode="anchor", fontsize=x_tick_fontsize, rotation=x_tick_rotation)
    ################################################ y ###############################################
    # 常规设置y轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    ax.set_ylabel(y_label_name, fontsize=y_label_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize, rotation=y_tick_rotation)
    # y轴可以超过一个值，但是tick最多只显示到该值
    if y_max_tick_to_one:
        ax.set_yticks([i for i in ax.get_yticks() if i <= y_max_tick_to_value])
    # y轴设置科学计数法
    if math_text:
        if np.min(data[i]) < 1 or np.max(data[i]) > 10:  # y轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.yaxis.set_major_formatter(formatter)
    # 设置y轴tick保留1位小数，会与“y轴设置科学计数法”冲突
    if one_decimal_place:
        if math_text:
            print('“one_decimal_place”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置y轴为百分数的显示效果，会与“y轴设置科学计数法”冲突
    if percentage:
        if math_text:
            print('“percentage”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))# 设置y轴为百分数的显示效果
    ###################################### 满足黄金比例的y轴设置 #####################################
    all_data_ymax = []
    all_data_ymin = []
    for i in range(0,len(labels_name)):
        all_data_ymax.append(np.max(eval('data'+str(i))))
        all_data_ymin.append(np.min(eval('data'+str(i))))
    y_max_value = np.max(all_data_ymax)
    y_min_value = np.min(all_data_ymin)
    y_max_min = y_max_value - y_min_value
    ax_min = y_min_value-(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max = y_max_value+(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max_y_max_value = ax_max - y_max_value
    # 如果y轴最小值不需要设置成0，则设置为黄金比例
    if y_lim_range is not None:
        ax.set_ylim(y_lim_range[0], y_lim_range[1])
    else:
        if ax_min_is_0:
            ax.set_ylim(0, ax_max)
        else:
            ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 ############################################
    if statistic:
        t_count = 0
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if test_method == 'ttest_ind':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_ind(data[i1], data[i2])
                elif test_method == 'ttest_rel':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_rel(data[i1], data[i2])
                elif test_method == 'mannwhitneyu':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.mannwhitneyu(data[i1], data[i2], alternative='two-sided')
                elif test_method == 'permutation_mean':
                    def per_statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    res = stats.permutation_test((data[i1], data[i2]), per_statistic, permutation_type='independent', n_resamples=10000, alternative='two-sided')
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                elif test_method == 'permutation_median':
                    def per_statistic(x, y):
                        return np.median(x) - np.median(y)
                    res = stats.permutation_test((data[i1], data[i2]), per_statistic, permutation_type='independent', n_resamples=10000, alternative='two-sided')
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                else:
                    print('没有该统计方法，请重新输入！！！')
                if multicorrect == True:
                    globals()['t_' + str(i1) + '_' + str(i2)] = eval('t_' + str(i1) + '_' + str(i2)) * (len(labels_name) * (len(labels_name) - 1)) / 2  # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    print('{} 方法，{} 和 {} 之间显著，s = {:.4f}，p = {:.4f}'.format(test_method, labels_name[i1], labels_name[i2], eval('s_' + str(i1) + '_' + str(i2)), eval('t_' + str(i1) + '_' + str(i2))))
        lines_interval = ax_max_y_max_value / (t_count + 1)
        star_line_interval = lines_interval / 5
        count = 1
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    ax.annotate('', xy=(i1 + 0.05, y_max_value + count * lines_interval), \
                                xytext=(i2 - 0.05, y_max_value + count * lines_interval), \
                                arrowprops=dict(edgecolor='0.5', width=0.5, headwidth=0.1, headlength=0.1))
                    if 0.01 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '*', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif 0.001 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.01:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '**', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif eval('t_' + str(i1) + '_' + str(i2)) <= 0.001:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '***', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    count += 1
    return y_max_value, ax_max_y_max_value

def plot_one_group_violin_figure(data, ax=None, labels_name=None, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', width=0.5, colors=None, title_name='', title_fontsize=10, title_pad=20, x_label_name='', x_label_fontsize=10, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, y_lim_range=None, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, statistic=False, test_method='ttest_ind', asterisk_fontsize=10, multicorrect=False, **kwargs):
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    if labels_name is None:
        labels_name = [str(i) for i in range(len(data))]
    if colors is None:
        colors = ['k'] * len(data)
    ##################################################################################################
    violin_data, violin_position = [], []
    for i in range(0, len(data)):
        violin_data.append(data[i])
        violin_position.append(i)
    x = np.arange(len(data))
    ######################################### 小提琴的各种参数 ########################################
    violin_parts = ax.violinplot(violin_data, positions=violin_position, widths=width, showmeans=False, showextrema=True, showmedians=False)
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)  # 控制小提琴面板颜色的alpha值
    violin_parts['cmaxes'].set_edgecolor('black')
    violin_parts['cmaxes'].set_alpha(1)
    violin_parts['cmins'].set_edgecolor('black')
    violin_parts['cmins'].set_alpha(1)
    violin_parts['cbars'].set_edgecolor('black')
    violin_parts['cbars'].set_alpha(1)
    quartile1, medians, quartile3 = [], [], []
    for each_group_data in violin_data:
        quartile1.append(np.percentile(each_group_data, 25))
        medians.append(np.percentile(each_group_data, 50))
        quartile3.append(np.percentile(each_group_data, 75))
    ax.scatter(violin_position, medians, marker='o', color='white', s=30, zorder=3)  # 中位数
    ax.vlines(violin_position, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ############################################### ax ###############################################
    ax.spines[['top', 'right']].set_visible(False)  # 去掉上边和右边的spine
    ############################################## title #############################################
    ax.set_title(title_name, fontsize=title_fontsize, pad=title_pad)
    ################################################ x ###############################################
    ax.set_xlabel(x_label_name, fontsize=x_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_name, ha=x_label_ha, rotation_mode="anchor", fontsize=x_tick_fontsize, rotation=x_tick_rotation)
    ################################################ y ###############################################
    # 常规设置y轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    ax.set_ylabel(y_label_name, fontsize=y_label_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize, rotation=y_tick_rotation)
    # y轴可以超过一个值，但是tick最多只显示到该值
    if y_max_tick_to_one:
        ax.set_yticks([i for i in ax.get_yticks() if i <= y_max_tick_to_value])
        ax.set_yticks([i for i in [0, 0.2, 0.4, 0.6, 0.8, 1] if i <= 1])
    # y轴设置科学计数法
    if math_text:
        if np.min(data[i]) < 1 or np.max(data[i]) > 10:  # y轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.yaxis.set_major_formatter(formatter)
    # 设置y轴tick保留1位小数，会与“y轴设置科学计数法”冲突
    if one_decimal_place:
        if math_text:
            print('“one_decimal_place”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置y轴为百分数的显示效果，会与“y轴设置科学计数法”冲突
    if percentage:
        if math_text:
            print('“percentage”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))# 设置y轴为百分数的显示效果
    ###################################### 满足黄金比例的y轴设置 #####################################
    all_data_ymax = []
    all_data_ymin = []
    for i in range(0,len(labels_name)):
        all_data_ymax.append(np.max(data[i]))
        all_data_ymin.append(np.min(data[i]))
    y_max_value = np.max(all_data_ymax)
    y_min_value = np.min(all_data_ymin)
    y_max_min = y_max_value - y_min_value
    ax_min = y_min_value-(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max = y_max_value+(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max_y_max_value = ax_max - y_max_value
    # 如果y轴最小值不需要设置成0，则设置为黄金比例
    if y_lim_range is not None:
        ax.set_ylim(y_lim_range[0], y_lim_range[1])
    else:
        if ax_min_is_0:
            ax.set_ylim(0, ax_max)
        else:
            ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 ############################################
    if statistic:
        t_count = 0
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if test_method == 'ttest_ind':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_ind(data[i1], data[i2])
                elif test_method == 'ttest_rel':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_rel(data[i1], data[i2])
                elif test_method == 'mannwhitneyu':
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.mannwhitneyu(data[i1], data[i2], alternative='two-sided')
                elif test_method == 'permutation_mean':
                    def per_statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    res = stats.permutation_test((data[i1], data[i2]), per_statistic, permutation_type='independent', n_resamples=10000, alternative='two-sided')
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                elif test_method == 'permutation_median':
                    def per_statistic(x, y):
                        return np.median(x) - np.median(y)
                    res = stats.permutation_test((data[i1], data[i2]), per_statistic, permutation_type='independent', n_resamples=10000, alternative='two-sided')
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                else:
                    print('没有该统计方法，请重新输入！！！')
                if multicorrect == True:
                    globals()['t_' + str(i1) + '_' + str(i2)] = eval('t_' + str(i1) + '_' + str(i2)) * (len(labels_name) * (len(labels_name) - 1)) / 2  # 多重比较校正，直接将p值乘以比较次数的bonferroni校正
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    print('{} 方法，{} 和 {} 之间显著，s = {:.4f}，p = {:.4f}'.format(test_method, labels_name[i1], labels_name[i2], eval('s_' + str(i1) + '_' + str(i2)), eval('t_' + str(i1) + '_' + str(i2))))
        ax_max_y_max_value = ax_max - y_max_value
        lines_interval = ax_max_y_max_value / (t_count + 1)
        star_line_interval = lines_interval / 5
        count = 1
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    ax.annotate('', xy=(i1 + 0.05, y_max_value + count * lines_interval), \
                                xytext=(i2 - 0.05, y_max_value + count * lines_interval), \
                                arrowprops=dict(edgecolor='0.5', width=0.5, headwidth=0.1, headlength=0.1))
                    if 0.01 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '*', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif 0.001 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.01:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '**', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif eval('t_' + str(i1) + '_' + str(i2)) <= 0.001:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '***', c='k', fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    count += 1
    return y_max_value, ax_max_y_max_value

def plot_correlation_figure(data1, data2, ax=None, stats_method='pearson', dots_color=None, line_color=None, title_name='', title_fontsize=10, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_major_locator=None, x_max_tick_to_one=False, x_max_tick_to_value=1, x_math_text=True, x_one_decimal_place=False, x_percentage=False, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_major_locator=None, y_max_tick_to_one=False, y_max_tick_to_value=1, y_math_text=True, y_one_decimal_place=False, y_percentage=False, asterisk_fontsize=10, summary=False):
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    ##################################################################################################
    exog, endog = sm.add_constant(data1), data2
    model = sm.OLS(endog, exog, missing="drop")
    results = model.fit()
    if summary :
        print(results.summary())
    data2_pred = results.predict(exog)  # 模型预测
    data1_idx = data1.argsort()  # 数组，[最小值的idx, 倒数第2小的值的idx, 倒数第3小的值的idx, ... , 最大值的idx]
    data1_ord = data1[data1_idx]  # 根据idx排序的新数组x_ord
    data2_pred_ord = data2_pred[data1_idx]
    # 画图
    ax.scatter(data1, data2, alpha=0.1, color=dots_color)
    ax.plot(data1_ord, data2_pred_ord, color=line_color)
    # ax.fill_between(x, y_est - y_err, y_est + y_err, color=line_color , alpha=0.4)
    ############################################### ax ###############################################
    ax.spines[['top', 'right']].set_visible(False)  # 去掉上边和右边的spine
    ############################################## title #############################################
    ax.set_title(title_name, fontsize=title_fontsize, pad=title_pad)
    ################################################ x ###############################################
    # 常规设置x轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    if x_major_locator is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_locator))  # 手动设置x轴tick之间的间隔
    ax.set_xlabel(x_label_name, fontsize=x_label_fontsize)
    ax.tick_params(axis='x', which='major', labelsize=x_tick_fontsize, rotation=x_tick_rotation)
    # x轴可以超过一个值，但是tick最多只显示到该值
    if x_max_tick_to_one:
        ax.set_xticks([i for i in ax.get_xticks() if i <= x_max_tick_to_value])
    # x轴设置科学计数法
    if x_math_text:
        if np.min(data1) < 0.01 or np.max(data1) > 100:  # x轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.xaxis.set_major_formatter(formatter)
    # 设置x轴tick保留1位小数，会与“x轴设置科学计数法”冲突
    if x_one_decimal_place:
        if x_math_text:
            print('“x_one_decimal_place”会与“x_math_text”冲突，请关闭“x_math_text”后再开启！')
        else:
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置x轴为百分数的显示效果，会与“x轴设置科学计数法”冲突
    if x_percentage:
        if x_math_text:
            print('“x_percentage”会与“x_math_text”冲突，请关闭“x_math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))# 设置y轴为百分数的显示效果
    ################################################ y ###############################################
    # 常规设置y轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    if y_major_locator is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_locator))  # 手动设置y轴tick之间的间隔
    ax.set_ylabel(y_label_name, fontsize=y_label_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize, rotation=y_tick_rotation)
    # y轴可以超过一个值，但是tick最多只显示到该值
    if y_max_tick_to_one:
        ax.set_yticks([i for i in ax.get_yticks() if i <= y_max_tick_to_value])
    # y轴设置科学计数法
    if y_math_text:
        if np.min(data1) < 0.01 or np.max(data1) > 100:  # y轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.yaxis.set_major_formatter(formatter)
    # 设置y轴tick保留1位小数，会与“y轴设置科学计数法”冲突
    if y_one_decimal_place:
        if y_math_text:
            print('“y_one_decimal_place”会与“y_math_text”冲突，请关闭“y_math_text”后再开启！')
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置y轴为百分数的显示效果，会与“y轴设置科学计数法”冲突
    if y_percentage:
        if y_math_text:
            print('“y_percentage”会与“y_math_text”冲突，请关闭“y_math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))# 设置y轴为百分数的显示效果
    ######################################## 标注r值和p值/星号 ########################################
    if stats_method == 'spearman':
        s, p = stats.spearmanr(data1, data2)
        r_or_rho=r'$\rho$'
        # print('斯皮尔曼相关性，r={}，p={}'.format(round(s, 3), round(p, 3)))
    elif stats_method == 'pearson':
        s, p = stats.pearsonr(data1, data2)
        r_or_rho='r'
        # print('皮尔逊相关性，r={}，p={}'.format(round(s, 3), round(p, 3)))
    x_start, x_end = ax.get_xlim()
    x_width = x_end - x_start
    y_start, y_end = ax.get_ylim()
    y_width = y_end - y_start
    if 0.01 < p < 0.05:
        asterisk_text = ' *'
    elif 0.001 < p < 0.01:
        asterisk_text = ' **'
    elif p < 0.001:
        asterisk_text = ' ***'
    else:
        asterisk_text = ''
    ax.text(x_start + x_width/ 10, y_start + 9 * y_width / 10, f'{r_or_rho}={str(round(s, 3))} {asterisk_text}', va='center', fontsize=asterisk_fontsize)  # 参数保留小数点后3位
    return

def plot_matrix_figure(data, ax=None, row_labels_name=[], col_labels_name=[], cmap='bwr', colorbar_label_name='', colorbar_label_pad=0.1, colorbar_tick_fontsize=10, colorbar_tick_rotation=0, x_rotation=60, row_labels_fontsize=5, col_labels_fontsize=5, colorbar=True,  colorbar_label_fontsize=10, title_name='', title_fontsize=15, title_pad=20, vmax=None, vmin=None, aspect='equal', **kwargs):
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    # 指定最大最小值
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    # 画图
    im = ax.imshow(data, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title_name, fontsize=title_fontsize, pad=title_pad)
    # 根据矩阵文件读取获得x、y轴长度以及标上label，并定义xy的labels字体大小
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels_name)
    ax.set_xticklabels(col_labels_name, fontsize=col_labels_fontsize)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels_name)
    ax.set_yticklabels(row_labels_name, fontsize=row_labels_fontsize)
    if colorbar:
        # 获取当前图形的坐标轴
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=colorbar_label_pad)  # pad 控制 colorbar 与图的距离
        # 创建colorbar
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(colorbar_label_name, rotation=-90, va="bottom", fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(axis='y', which='major', labelsize=colorbar_tick_fontsize, rotation=colorbar_tick_rotation)
        # 调整 colorbar 的高度，使其与图的高度相同
        ax_height = ax.get_position().height
        cax.set_position([cax.get_position().x0, cax.get_position().y0, cax.get_position().width, ax_height])
    # 旋转x轴label并对齐到每个cell中线
    plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha="right", rotation_mode="anchor")  # 其中“right”表示label最右边字母对齐到每个cell的中线
    return

def plot_human_brain_figure(data, surf='veryinflated', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name='', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False):
    # 设置必要文件路径
    current_dir = os.path.dirname(__file__)
    neuromaps_data_dir = op.join(current_dir, 'neuromaps-data')
    lh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.L.Glasser.32k_fs_LR.label.gii')
    rh_Glasser_atlas_dir = op.join(current_dir, 'neuromaps-data', 'atlases', 'rjx_Glasser_atlas', 'fsaverage.R.Glasser.32k_fs_LR.label.gii')
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
    # 加载Glasser分区数据
    df = pd.read_csv(op.join(current_dir, 'human_glasser.csv'))
    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    lh_parc = nib.load(lh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in lh_roi_list}
    for index, roi_index in enumerate(lh_parc):
        if roi_index-181 >= 0:
            roi_vertics[lh_roi_list[roi_index-181]].append(index)
    lh_parc = np.zeros_like(lh_parc)
    for roi_data in lh_data:
        lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    rh_parc = nib.load(rh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in rh_roi_list}
    for index, roi_index in enumerate(rh_parc):
        if roi_index-1 >= 0:
            roi_vertics[rh_roi_list[roi_index-1]].append(index)
    rh_parc = np.zeros_like(rh_parc)
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
    return fig

def plot_human_hemi_brain_figure(data, hemi='lh', surf='veryinflated', vmin=None, vmax=None, cmap='Reds', colorbar=True, colorbar_location='right', colorbar_label_name= '', colorbar_label_rotation=0, colorbar_decimals=1, colorbar_fontsize=8, colorbar_nticks=2, colorbar_shrink=0.15, colorbar_aspect=8, colorbar_draw_border=False):
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
    df = pd.read_csv(op.join(current_dir, 'human_glasser.csv'))
    lh_roi_list, rh_roi_list = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    lh_parc = nib.load(lh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in lh_roi_list}
    for index, roi_index in enumerate(lh_parc):
        if roi_index-181 >= 0:
            roi_vertics[lh_roi_list[roi_index-181]].append(index)
    lh_parc = np.zeros_like(lh_parc)
    for roi_data in lh_data:
        lh_parc[roi_vertics[roi_data]] = lh_data[roi_data]
    rh_parc = nib.load(rh_Glasser_atlas_dir).darrays[0].data
    roi_vertics = {roi:[] for roi in rh_roi_list}
    for index, roi_index in enumerate(rh_parc):
        if roi_index-1 >= 0:
            roi_vertics[rh_roi_list[roi_index-1]].append(index)
    rh_parc = np.zeros_like(rh_parc)
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
    return fig

def plot_macaque_brain_figure(data, cmap='Reds', surf='inflated', vmin=None, vmax=None, colorbar=True, colorbar_direction='vertical', colorbar_label_name='', colorbar_label_fontsize=10, colorbar_tick_fontsize=10, colorbar_tick_rotation=0, colorbar_outline=False):
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subj_dir = op.join(current_dir, 'FS')
    subj = "NMT"
    lh_atlas_path, rh_atlas_path = op.join(subj_dir, subj, 'label', 'L.charm5.label.gii'), op.join(subj_dir, subj, 'label', 'L.charm5.label.gii')  # 对称图集，两边最大label都是88
    # 默认参数
    if not isinstance(data, dict):
        df = pd.read_csv(data)
        data = {key:value for key,value in zip(df["ROIs_name"], df["Values"])}
    if not vmin:
        vmin = np.min(list(data.values()))
    if not vmax:
        vmax = np.max(list(data.values()))
    if vmin == vmax and vmin > 0:
        vmin = 0
    elif vmin == vmax and vmax < 0:
        vmax = 0

    # 读取图集数据
    lh_atlas, rh_atlas = nib.load(lh_atlas_path), nib.load(rh_atlas_path)
    lh_atlas_data, rh_atlas_data = lh_atlas.darrays[0].data, rh_atlas.darrays[0].data
    df = pd.read_csv(op.join(current_dir, 'macaque_charm5.csv'))
    lh_rois_name, rh_rois_name = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    lh_label_roi, rh_label_roi = {index+1:roi for index, roi in enumerate(lh_rois_name)}, {index+1:roi for index, roi in enumerate(rh_rois_name)}  # {1: 'lh_area_32', 2: 'lh_area_25', ...}
    # 转换Overlay数据
    lh_index_label = {}
    for index, label in enumerate(lh_atlas_data):
        lh_index_label[index] = label
    lh_plot_data = np.zeros(lh_atlas_data.shape)
    for index in range(lh_plot_data.shape[0]):
        label = lh_index_label[index]
        if label == 0:  # 如果label为0，跳过。因为没有为0分配脑区名字
            continue
        roi = lh_label_roi[label]
        if roi in data:  # 如果roi不在data中，则默认值为0
            value = data[roi]
        else:
            value = 0
        lh_plot_data[index] = value
    rh_index_label = {}
    for index, label in enumerate(rh_atlas_data):
        rh_index_label[index] = label
    rh_plot_data = np.zeros(rh_atlas_data.shape)
    for index in range(rh_plot_data.shape[0]):
        label = rh_index_label[index]
        if label == 0:  # 如果label为0，跳过。因为没有为0分配脑区名字
            continue
        roi = rh_label_roi[label]
        if roi in data:  # 如果roi不在data中，则默认值为0
            value = data[roi]
        else:
            value = 0
        rh_plot_data[index] = value
    # 画图
    ax_args = {'hemi':['lh', 'rh', 'lh', 'rh'], 'view':['lateral', 'lateral', 'medial', 'medial'], 'plot_data':[lh_plot_data, rh_plot_data, lh_plot_data, rh_plot_data]}  # 'rostral', 'caudal', 'dorsal', 'ventral', 'lateral', 'medial', 'frontal', 'parietal'
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for index, ax in enumerate(axes.flat):
        # 画Underlay
        brain = mne.viz.Brain(subj, hemi=ax_args['hemi'][index], surf=surf, subjects_dir=subj_dir, cortex="low_contrast", background="white", views=ax_args['view'][index], size=[1800,1000])
        # 画Overlay
        plot_data = ax_args['plot_data'][index]
        if np.all(plot_data == 0) or np.all(np.isnan(plot_data)):
            fmin, fmax = -1, 1  # 当数据全部为0时，需要设置fmin和fmax，否则会报错
        else:
            fmin, fmax = vmin, vmax
        plot_data = plot_data.astype(float)
        plot_data[plot_data == 0] =np.nan  # 将0值成nan值，可以保证没有值的脑区不被分配任何颜色
        brain.add_data(plot_data, colormap=cmap, colorbar=False, fmin=fmin, fmax=fmax)
        screenshot = brain.screenshot()
        # 把sreenshot截到最小
        for row in range(screenshot.shape[0]):
            if np.mean(screenshot[row,:,:]) != 255:
                row1 = row - 1
                break
        for row in range(0,-screenshot.shape[0], -1):
            if np.mean(screenshot[row,:,:]) != 255:
                row2 = screenshot.shape[0] + row
                break
        for col in range(screenshot.shape[1]):
            if np.mean(screenshot[:,col,:]) != 255:
                col1 = col - 1
                break
        for col in range(0,-screenshot.shape[1], -1):
            if np.mean(screenshot[:,col,:]) != 255:
                col2 = screenshot.shape[1] + col
                break
        screenshot = screenshot[row1:row2, col1:col2, :]
        brain.close()
        im = ax.imshow(screenshot)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ############################################### colorbar ###############################################
    sm = ScalarMappable(cmap=cmap)
    sm.set_array((vmin, vmax))  # 设置值范围
    if colorbar:
        formatter = ScalarFormatter(useMathText=True)  # 科学计数法相关
        formatter.set_powerlimits((-1, 2))  # <=-1也就是小于等于0.1，>=2，也就是大于等于100，会写成科学计数法
        if colorbar_direction == 'vertical':
            cax = fig.add_axes([1, 0.425, 0.01, 0.15])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', cmap=cmap)  # "vertical", "horizontal"
            cbar.ax.set_ylabel(colorbar_label_name, fontsize=10)
            cbar.ax.yaxis.set_label_position("left")  # 原本设置y轴label默认在右边，现在换到左边
            cbar.ax.tick_params(axis='y', which='major', labelsize=colorbar_tick_fontsize, rotation=colorbar_tick_rotation, length=0)
            if vmax < 0.1 or vmax > 100:  # y轴设置科学计数法
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.set_offset_position('left')
                cbar.ax.yaxis.get_offset_text().set_y(2)
                cbar.ax.yaxis.get_offset_text().set_position((5, 0))  # 貌似set_y不起作用，只能靠set_x来排版
        elif colorbar_direction == 'horizontal':
            cax = fig.add_axes([0.44, 0.53, 0.15, 0.01])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', cmap=cmap)  # "vertical", "horizontal"
            cbar.ax.set_title(colorbar_label_name, fontsize=colorbar_label_fontsize)
            cbar.ax.tick_params(axis='x', which='major', labelsize=colorbar_tick_fontsize, rotation=colorbar_tick_rotation, length=0)
            if vmax < 0.1 or vmax > 100:  # y轴设置科学计数法
                cbar.ax.xaxis.set_major_formatter(formatter)
        if not colorbar_outline:
            cbar.outline.set_visible(False)  # 去除colorbar的边框
        cbar.set_ticks([vmin, vmax])
    return fig

def plot_macaque_hemi_brain_figure(data, ax_direction='horizontal', hemi='lh', cmap='Reds', surf='inflated', vmin=None, vmax=None, colorbar=True, colorbar_direction='vertical', colorbar_label_name='', colorbar_label_fontsize=10, colorbar_tick_fontsize=10, colorbar_tick_rotation=0, colorbar_outline=False):
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subj_dir = op.join(current_dir, 'FS')
    subj = "NMT"
    lh_atlas_path, rh_atlas_path = op.join(subj_dir, subj, 'label', 'L.charm5.label.gii'), op.join(subj_dir, subj, 'label', 'L.charm5.label.gii')  # 对称图集，两边最大label都是88
    # 默认参数
    if not isinstance(data, dict):
        df = pd.read_csv(data)
        data = {key:value for key,value in zip(df["ROIs_name"], df["Values"])}
    if not vmin:
        vmin = np.min(list(data.values()))
    if not vmax:
        vmax = np.max(list(data.values()))
    if vmin == vmax and vmin > 0:
        vmin = 0
    elif vmin == vmax and vmax < 0:
        vmax = 0
    # 读取图集数据
    lh_atlas, rh_atlas = nib.load(lh_atlas_path), nib.load(rh_atlas_path)
    lh_atlas_data, rh_atlas_data = lh_atlas.darrays[0].data, rh_atlas.darrays[0].data
    df = pd.read_csv(op.join(current_dir, 'macaque_charm5.csv'))
    lh_rois_name, rh_rois_name = list(df['ROIs_name'])[0: int(len(df['ROIs_name'])/2)], list(df['ROIs_name'])[int(len(df['ROIs_name'])/2): len(df['ROIs_name'])]
    lh_label_roi, rh_label_roi = {index+1:roi for index, roi in enumerate(lh_rois_name)}, {index+1:roi for index, roi in enumerate(rh_rois_name)}  # {1: 'lh_area_32', 2: 'lh_area_25', ...}
    # 转换Overlay数据
    if hemi == 'lh':
        lh_index_label = {}
        for index, label in enumerate(lh_atlas_data):
            lh_index_label[index] = label
        lh_plot_data = np.zeros(lh_atlas_data.shape)
        for index in range(lh_plot_data.shape[0]):
            label = lh_index_label[index]
            if label == 0:  # 如果label为0，跳过。因为没有为0分配脑区名字
                continue
            roi = lh_label_roi[label]
            if roi in data:  # 如果roi不在data中，则默认值为0
                value = data[roi]
            else:
                value = 0
            lh_plot_data[index] = value
            plot_data = lh_plot_data
    elif hemi == 'rh':
        rh_index_label = {}
        for index, label in enumerate(rh_atlas_data):
            rh_index_label[index] = label
        rh_plot_data = np.zeros(rh_atlas_data.shape)
        for index in range(rh_plot_data.shape[0]):
            label = rh_index_label[index]
            if label == 0:  # 如果label为0，跳过。因为没有为0分配脑区名字
                continue
            roi = rh_label_roi[label]
            if roi in data:  # 如果roi不在data中，则默认值为0
                value = data[roi]
            else:
                value = 0
            rh_plot_data[index] = value
            plot_data = rh_plot_data
    # 画图
    ax_args = {'view':['lateral','medial']}  # 'rostral', 'caudal', 'dorsal', 'ventral', 'lateral', 'medial', 'frontal', 'parietal'
    if ax_direction == 'horizontal':
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    elif ax_direction == 'vertical':
        fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for index, ax in enumerate(axes.flat):
        # 画Underlay
        brain = mne.viz.Brain(subj, hemi=hemi, surf=surf, subjects_dir=subj_dir, cortex="low_contrast", background="white", views=ax_args['view'][index], size=[1800,1000])
        # 画Overlay
        if np.all(plot_data == 0) or np.all(np.isnan(plot_data)):
            fmin, fmax = -1, 1  # 当数据全部为0时，需要设置fmin和fmax，否则会报错
        else:
            fmin, fmax = vmin, vmax
        plot_data = plot_data.astype(float)
        plot_data[plot_data == 0] = np.nan  # 将0值成nan值，可以保证没有值的脑区不被分配任何颜色
        brain.add_data(plot_data, colormap=cmap, colorbar=False, fmin=fmin, fmax=fmax)
        screenshot = brain.screenshot()
        # 把sreenshot截到最小
        for row in range(screenshot.shape[0]):
            if np.mean(screenshot[row,:,:]) != 255:
                row1 = row - 1
                break
        for row in range(0,-screenshot.shape[0], -1):
            if np.mean(screenshot[row,:,:]) != 255:
                row2 = screenshot.shape[0] + row
                break
        for col in range(screenshot.shape[1]):
            if np.mean(screenshot[:,col,:]) != 255:
                col1 = col - 1
                break
        for col in range(0,-screenshot.shape[1], -1):
            if np.mean(screenshot[:,col,:]) != 255:
                col2 = screenshot.shape[1] + col
                break
        screenshot = screenshot[row1:row2, col1:col2, :]
        brain.close()
        im = ax.imshow(screenshot)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ############################################### colorbar ###############################################
    sm = ScalarMappable(cmap=cmap)
    sm.set_array((vmin, vmax))  # 设置值范围
    if colorbar:
        formatter = ScalarFormatter(useMathText=True)  # 科学计数法相关
        formatter.set_powerlimits((-1, 2))  # <=-1也就是小于等于0.1，>=2，也就是大于等于100，会写成科学计数法
        if colorbar_direction == 'vertical':
            if ax_direction == 'horizontal':
                cax = fig.add_axes([1, 0.425, 0.01, 0.15])  # [left, bottom, width, height]
            elif ax_direction == 'vertical':
                cax = fig.add_axes([0.8, 0.425, 0.01, 0.15])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', cmap=cmap)  # "vertical", "horizontal"
            cbar.ax.set_ylabel(colorbar_label_name, fontsize=colorbar_label_fontsize)
            cbar.ax.yaxis.set_label_position("left")  # 原本设置y轴label默认在右边，现在换到左边
            cbar.ax.tick_params(axis='y', which='major', labelsize=colorbar_tick_fontsize, rotation=colorbar_tick_rotation, length=0)
            if vmax < 0.1 or vmax > 100:  # y轴设置科学计数法
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.set_offset_position('left')
                cbar.ax.yaxis.get_offset_text().set_y(2)
                cbar.ax.yaxis.get_offset_text().set_position((5, 0))  # 貌似set_y不起作用，只能靠set_x来排版
        elif colorbar_direction == 'horizontal':
            if ax_direction == 'horizontal':
                cax = fig.add_axes([0.44, 0.3, 0.15, 0.01])  # [left, bottom, width, height]
            elif ax_direction == 'vertical':
                cax = fig.add_axes([0.44, 0, 0.15, 0.01])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', cmap=cmap)  # "vertical", "horizontal"
            cbar.ax.set_title(colorbar_label_name, fontsize=colorbar_label_fontsize)
            cbar.ax.tick_params(axis='x', which='major', labelsize=colorbar_tick_fontsize, rotation=colorbar_tick_rotation, length=0)
            if vmax < 0.1 or vmax > 100:  # y轴设置科学计数法
                    cbar.ax.xaxis.set_major_formatter(formatter)
        if not colorbar_outline:
            cbar.outline.set_visible(False)  # 去除colorbar的边框
        cbar.set_ticks([vmin, vmax])
    return fig

def plot_multi_group_bar_figure(data, test_method, legend_name=None, labels_name=None, ax=None, width=0.2, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=30, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, math_text=True, y_max_tick_to_one=False, y_max_tick_to_value=1, one_decimal_place=False, percentage=False, legend_fontsize=10, legend_location='best', legend_bbox_location=None, ax_min_is_0=False, asterisk_size=10, multicorrect=False, tails='two-sided', **kwargs):
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    if legend_name is None:
        legend_name = [str(i) for i in range(len(data))]
    if labels_name is None:
        labels_name = [str(i) for i in range(data[0].shape[1])]
    if colors is None:
        colors = ['k'] * len(data)
    ##################################################################################################
    for i1 in range(0,len(legend_name)):  # i1 1~2
        all_bar_mean = []
        all_bar_SD = []
        all_bar_len = []
        for i2 in range(0,len(labels_name)):  # i2 1~3
            globals()['data'+str(i1)+'_bar'+str(i2)] = data[i1][:,i2]
            all_bar_mean.append(np.mean(eval('data'+str(i1)+'_bar'+str(i2))))
            all_bar_SD.append(np.std(eval('data'+str(i1)+'_bar'+str(i2))))
            all_bar_len.append(len(eval('data'+str(i1)+'_bar'+str(i2))))
        globals()['data'+str(i1)+'_mean'] = np.array(all_bar_mean)
        globals()['data'+str(i1)+'_SD'] = np.array(all_bar_SD)
        globals()['data'+str(i1)+'_len'] = np.array(all_bar_len)
        globals()['data'+str(i1)+'_SE'] = eval('data'+str(i1)+'_SD') / eval('data'+str(i1)+'_len') ** 0.5
    x = np.arange(len(labels_name))
    position = []
    for i in range(0,len(legend_name)):
        globals()['bar'+str(i)] = ax.bar(x - 0.4 + (2 * i + 1) * 0.8 / (2 * len(legend_name)), eval('data'+str(i)+'_mean'), width=width, color=colors[i])
        position.append(x - 0.4 + (2 * i + 1) * 0.8 / (2 * len(legend_name)))
        ax.errorbar(x - 0.4 + (2 * i + 1) * 0.8 / (2 * len(legend_name)), eval('data'+str(i)+'_mean'), eval('data'+str(i)+'_SE'), fmt='none', linewidth=1, capsize=3, color='black')
    ############################################### ax ###############################################
    ax.spines[['top', 'right']].set_visible(False)  # 去掉上边和右边的spine
    ############################################## title #############################################
    ax.set_title(title_name,fontsize=title_fontsize, pad=title_pad)
    ################################################ x ###############################################
    ax.set_xlabel(x_label_name,fontsize=x_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_name,ha="right",rotation_mode="anchor",fontsize=x_tick_fontsize,rotation=x_tick_rotation)
    ################################################ y ###############################################
    # 常规设置y轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    ax.set_ylabel(y_label_name,fontsize=y_label_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize, rotation=y_tick_rotation)
    # y轴可以超过一个值，但是tick最多只显示到该值
    if y_max_tick_to_one:
        ax.set_yticks([i for i in ax.get_yticks() if i <= y_max_tick_to_value])
    # y轴设置科学计数法
    if math_text:
        if np.min(all_bar_mean) < 0.01 or np.max(all_bar_mean) > 100:  # y轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.yaxis.set_major_formatter(formatter)
    # 设置y轴tick保留1位小数，会与“y轴设置科学计数法”冲突
    if one_decimal_place:
        if math_text:
            print('“one_decimal_place”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置y轴为百分数的显示效果，会与“y轴设置科学计数法”冲突
    if percentage:
        if math_text:
            print('“percentage”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))  # 设置y轴为百分数的显示效果
    ############################################# legends ############################################
    all_bars = []
    for i in range(0,len(legend_name)):
        all_bars.append(eval('bar'+str(i)))
    if legend_bbox_location is None:
        ax.legend(all_bars,legend_name, loc=legend_location, fontsize=legend_fontsize)
    else:
        ax.legend(all_bars, legend_name, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_location)
    ###################################### 满足黄金比例的xy轴设置 #####################################
    all_data_ymax = []
    all_data_ymin = []
    for i in range(0,len(legend_name)):
        all_data_ymax.append(eval('data'+str(i)+'_mean') + eval('data'+str(i)+'_SE'))
        all_data_ymin.append(eval('data'+str(i)+'_mean') - eval('data'+str(i)+'_SE'))
    y_max_value = np.max(all_data_ymax)
    y_min_value = np.min(all_data_ymin)
    y_max_min = y_max_value - y_min_value
    ax_min = y_min_value-(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max = y_max_value+(y_max_min/(5**0.5-1)-y_max_min/2)
    # 如果y轴最小值不需要设置成0，则设置为黄金比例
    if ax_min_is_0:
        ax.set_ylim(0, ax_max)
    else:
        ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 #############################################
    t_count_list = []
    for i0 in range(len(labels_name)):
        t_count = 0
        for i1 in range(len(legend_name)):
            for i2 in range(i1 + 1, len(legend_name)):
                if test_method == 'ttest_ind':
                    globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.ttest_ind(data[i1][:,i0], data[i2][:,i0])  # 独立样本t检验
                elif test_method == 'ttest_rel':
                    globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.ttest_rel(data[i1][:,i0], data[i2][:,i0])  # 配对样本t检验
                elif test_method == 'mannwhitneyu':
                    globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.mannwhitneyu(data[i1][:,i0], data[i2][:,i0], alternative='two-sided')  # Mann Whitney Wilcoxon秩和检验
                elif test_method == 'permutation_mean':
                    def per_statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    res = stats.permutation_test((data[i1][:,i0], data[i2][:,i0]), per_statistic, permutation_type='independent', n_resamples=10000, alternative=tails)
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                elif test_method == 'permutation_median':
                    def per_statistic(x, y):
                        return np.median(x) - np.median(y)
                    res = stats.permutation_test((data[i1][:,i0], data[i2][:,i0]), per_statistic, permutation_type='independent', n_resamples=10000, alternative=tails)
                    globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                else:
                    print('没有该统计方法，请重新输入！！！')
                if multicorrect==True:
                    globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) * len(labels_name) * (len(legend_name) * (len(legend_name) - 1)) / 2   # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    # print('{} 方法，{} 组里的 {} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i0], legend_name[i1], legend_name[i2], round(eval('s' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3), round(eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3)))
        t_count_list.append(t_count)
    t_count = np.max(t_count_list)
    ax_max_y_max_value = ax_max - y_max_value
    lines_interval = ax_max_y_max_value / (t_count + 1)
    star_line_interval = lines_interval / 5
    position = np.array(position).T
    for i0 in range(len(labels_name)):
        count = 1
        for i1 in range(len(legend_name)):
            for i2 in range(i1 + 1, len(legend_name)):
                if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                    ax.annotate('', xy=(position[i0, i1] + 0.01, y_max_value + count * lines_interval), \
                            xytext=(position[i0, i2] - 0.01, y_max_value + count * lines_interval), \
                            arrowprops=dict(edgecolor='0.5', width=0.5, headwidth=0.1, headlength=0.1))
                    if 0.01 < eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                        ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                            '*', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                    elif 0.001 < eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.01:
                        ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                            '**', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                    elif eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.001:
                        ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                            '***', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                    count += 1
    return

def plot_multi_group_violin_figure(data, test_method, legend_name=None, labels_name=None, ax=None, width=0.2, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_tick_ha='center', y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, math_text=True, y_max_tick_to_one=False, y_max_tick_to_value=1, one_decimal_place=False, percentage=False, legend_fontsize=10, legend_location='best', legend_bbox_location=None, ax_min_is_0=False, asterisk_size=10, statistic=True, multicorrect=False, tails='two-sided', **kwargs):
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    if legend_name is None:
        legend_name = [str(i) for i in range(len(data))]
    if labels_name is None:
        labels_name = [str(i) for i in range(len(data[0]))]
    if colors is None:
        colors = ['k'] * len(data)
    ##################################################################################################
    x = np.arange(len(labels_name))
    position = []
    for i in range(len(legend_name)):  # 总共只有0和1这 两个值
        violin_data = data[i]  # data[0]是一个list，里面有3个元素
        violin_position = x - 0.4 + (2 * i + 1) * 0.8 / (2 * len(legend_name))
        position.append(violin_position)
        globals()['bar'+str(i)] = ax.bar(violin_position, 0, color=colors[i])  # legend还得靠ax.bar
        violin_parts = ax.violinplot(violin_data, positions=violin_position, widths=width, showmeans=False, showextrema=True, showmedians=False)
        ####################################### 小提琴的各种参数 ######################################
        # 'bodies', 'cmeans', 'cmaxes', 'cmins', 'cbars'
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.5)
        violin_parts['cmaxes'].set_edgecolor(colors[i])
        violin_parts['cmaxes'].set_alpha(1)
        violin_parts['cmins'].set_edgecolor(colors[i])
        violin_parts['cmins'].set_alpha(1)
        violin_parts['cbars'].set_edgecolor(colors[i])
        violin_parts['cbars'].set_alpha(1)
        quartile1, medians, quartile3 = [], [], []
        for i1 in range(len(data[i])):  # 遍历MST、MT、FST
            each_group_data = data[i][i1]
            quartile1.append(np.percentile(each_group_data, 25))
            medians.append(np.percentile(each_group_data, 50))
            quartile3.append(np.percentile(each_group_data, 75))
        ax.scatter(violin_position, medians, marker='o', color='white', s=30, zorder=3)  # 中位数
        ax.vlines(violin_position, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ############################################### ax ###############################################
    ax.spines[['top', 'right']].set_visible(False)  # 去掉上边和右边的spine
    ############################################## title #############################################
    ax.set_title(title_name,fontsize=title_fontsize, pad=title_pad)
    ################################################ x ###############################################
    ax.set_xlabel(x_label_name,fontsize=x_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_name,ha=x_tick_ha,rotation_mode="anchor",fontsize=x_tick_fontsize,rotation=x_tick_rotation)
    ################################################ y ###############################################
    # 常规设置y轴“label名字，label字体大小，tick字体大小，tick旋转角度”
    ax.set_ylabel(y_label_name,fontsize=y_label_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize, rotation=y_tick_rotation)
    # y轴可以超过一个值，但是tick最多只显示到该值
    if y_max_tick_to_one:
        ax.set_yticks([i for i in ax.get_yticks() if i <= y_max_tick_to_value])
    # y轴设置科学计数法
    if math_text:
        if np.min(list(chain.from_iterable(list(chain.from_iterable(data))))) < 0.01 or np.max(list(chain.from_iterable(list(chain.from_iterable(data))))) > 100:  # y轴设置科学计数法
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))  # <=-2也就是小于等于0.01，>=2，也就是大于等于100，会写成科学计数法
            ax.yaxis.set_major_formatter(formatter)
    # 设置y轴tick保留1位小数，会与“y轴设置科学计数法”冲突
    if one_decimal_place:
        if math_text:
            print('“one_decimal_place”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # 设置y轴为百分数的显示效果，会与“y轴设置科学计数法”冲突
    if percentage:
        if math_text:
            print('“percentage”会与“math_text”冲突，请关闭“math_text”后再开启！')
        else:
            def percentage_formatter(x, pos):# 设置y轴为百分数的显示效果
                # x: 坐标值, pos: 小数点位置
                return '{:.0%}'.format(x)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))  # 设置y轴为百分数的显示效果
    ############################################# legends ############################################
    all_bars = []
    for i in range(0,len(legend_name)):
        all_bars.append(eval('bar'+str(i)))
    if legend_bbox_location is None:
        ax.legend(all_bars,legend_name, loc=legend_location, fontsize=legend_fontsize)
    else:
        ax.legend(all_bars, legend_name, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_location)
    ###################################### 满足黄金比例的xy轴设置 #####################################
    all_data_ymax = []
    all_data_ymin = []
    for i in range(0, len(legend_name)):
        all_data_ymax.append(np.max(list(chain.from_iterable(list(chain.from_iterable(data))))))
        all_data_ymin.append(np.min(list(chain.from_iterable(list(chain.from_iterable(data))))))
    y_max_value = np.max(all_data_ymax)
    y_min_value = np.min(all_data_ymin)
    y_max_min = y_max_value - y_min_value
    ax_min = y_min_value-(y_max_min/(5**0.5-1)-y_max_min/2)
    ax_max = y_max_value+(y_max_min/(5**0.5-1)-y_max_min/2)
    # 如果y轴最小值不需要设置成0，则设置为黄金比例
    if ax_min_is_0:
        ax.set_ylim(0, ax_max)
    else:
        ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 #############################################
    if statistic:
        t_count_list = []
        for i0 in range(len(labels_name)):
            t_count = 0
            for i1 in range(len(legend_name)):
                for i2 in range(i1 + 1, len(legend_name)):
                    if test_method == 'ttest_ind':
                        globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.ttest_ind(data[i1][i0], data[i2][i0])
                    elif test_method == 'ttest_rel':
                        globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.ttest_rel(data[i1][i0], data[i2][i0])
                    elif test_method == 'mannwhitneyu':
                        globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                        stats.mannwhitneyu(data[i1][i0], data[i2][i0], alternative='two-sided')
                    elif test_method == 'permutation_mean':
                        def per_statistic(x, y):
                            return np.mean(x) - np.mean(y)
                        res = stats.permutation_test((data[i1][i0], data[i2][i0]), per_statistic, permutation_type='independent', n_resamples=10000, alternative=tails)
                        globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                    elif test_method == 'permutation_median':
                        def per_statistic(x, y):
                            return np.median(x) - np.median(y)
                        res = stats.permutation_test((data[i1][i0], data[i2][i0]), per_statistic, permutation_type='independent', n_resamples=10000, alternative=tails)
                        globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = res.statistic, res.pvalue
                    else:
                        print('没有该统计方法，请重新输入！！！')

                    if multicorrect == True:
                        globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) * len(labels_name) * (len(legend_name) * (len(legend_name) - 1)) / 2   # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                    if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                        t_count +=1
                        # print('{} 方法，{} 组里的 {} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i0], legend_name[i1], legend_name[i2], round(eval('s' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3), round(eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3)))
            t_count_list.append(t_count)
        t_count = np.max(t_count_list)
        ax_max_y_max_value = ax_max - y_max_value
        lines_interval = ax_max_y_max_value / (t_count + 1)
        star_line_interval = lines_interval / 5
        position = np.array(position).T
        for i0 in range(len(labels_name)):
            count = 1
            for i1 in range(len(legend_name)):
                for i2 in range(i1 + 1, len(legend_name)):
                    if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                        ax.annotate('', xy=(position[i0, i1] + 0.01, y_max_value + count * lines_interval), \
                                xytext=(position[i0, i2] - 0.01, y_max_value + count * lines_interval), \
                                arrowprops=dict(edgecolor='0.5', width=0.5, headwidth=0.1, headlength=0.1))
                        if 0.01 < eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                            ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '*', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                        elif 0.001 < eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.01:
                            ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '**', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                        elif eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.001:
                            ax.text((position[i0, i1] + position[i0, i2]) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '***', c='k', fontsize=asterisk_size, horizontalalignment='center', verticalalignment='center')
                        count += 1
    return