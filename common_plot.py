import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import multitest


def plot_one_group_bar_figure(data, ax=None, labels_name=None, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', width=0.5, colors=None, title_name='', title_fontsize=10, title_pad=20, x_label_name='', x_label_fontsize=10, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, y_lim_range=None, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, statistic=False, p_list=None, test_method='ttest_ind', asterisk_fontsize=10, asterisk_color='k', line_color='0.5', multicorrect_bonferroni=False, multicorrect_fdr=False, **kwargs):
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
        if np.min(data[i]) < 0.1 or np.max(data[i]) > 100:  # y轴设置科学计数法
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
        ax_max = y_lim_range[1]
        ax_max_y_max_value = ax_max - y_max_value
    else:
        if ax_min_is_0:
            ax.set_ylim(0, ax_max)
        else:
            ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 ############################################
    p_list_fdr = []
    if statistic:
        t_count = 0
        p_list_index = 0
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if test_method == 'external':  #############################################
                    globals()['t_' + str(i1) + '_' + str(i2)] = p_list[p_list_index]  #############################################
                    p_list_index += 1
                elif test_method == 'ttest_ind':
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
                if multicorrect_bonferroni == True:
                    globals()['t_' + str(i1) + '_' + str(i2)] = eval('t_' + str(i1) + '_' + str(i2)) * (len(labels_name) * (len(labels_name) - 1)) / 2  # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                if multicorrect_fdr == True:
                    p_list_fdr.append(eval('t_' + str(i1) + '_' + str(i2)))
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    # print('{} 方法，{} 和 {} 之间显著，s = {:.4f}，p = {:.4f}'.format(test_method, labels_name[i1], labels_name[i2], eval('s_' + str(i1) + '_' + str(i2)), eval('t_' + str(i1) + '_' + str(i2))))
        if multicorrect_fdr == True:
            _, p_list_fdr_corr = multitest.fdrcorrection(p_list_fdr, alpha=0.05, method='i', is_sorted=False)
            p_list_fdr_corr_index = 0
            for i1 in range(len(labels_name)):
                for i2 in range(i1+1, len(labels_name)):
                    globals()['t_' + str(i1) + '_' + str(i2)] = p_list_fdr_corr[p_list_fdr_corr_index]
                    p_list_fdr_corr_index += 1
        lines_interval = ax_max_y_max_value / (t_count + 1)
        star_line_interval = lines_interval / 5
        count = 1
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                    ax.annotate('', xy=(i1 + 0.05, y_max_value + count * lines_interval), \
                                xytext=(i2 - 0.05, y_max_value + count * lines_interval), \
                                arrowprops=dict(edgecolor=line_color, width=0.5, headwidth=0.1, headlength=0.1))
                    if 0.01 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '*', c=asterisk_color, fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif 0.001 < eval('t_' + str(i1) + '_' + str(i2)) <= 0.01:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '**', c=asterisk_color, fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    elif eval('t_' + str(i1) + '_' + str(i2)) <= 0.001:
                        ax.text((i1 + i2) / 2, y_max_value + count * lines_interval + star_line_interval, \
                                '***', c=asterisk_color, fontsize=asterisk_fontsize, horizontalalignment='center', verticalalignment='center')
                    count += 1
    return y_max_value, ax_max_y_max_value

def plot_one_group_violin_figure(data, ax=None, labels_name=None, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', width=0.5, colors=None, title_name='', title_fontsize=10, title_pad=20, x_label_name='', x_label_fontsize=10, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, y_lim_range=None, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, statistic=False, p_list=None, test_method='ttest_ind', asterisk_fontsize=10, multicorrect=False, **kwargs):
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
        p_list_index = 0
        for i1 in range(len(labels_name)):
            for i2 in range(i1+1, len(labels_name)):
                if  test_method == 'external':  #############################################
                    globals()['t_' + str(i1) + '_' + str(i2)] = p_list[p_list_index]  #############################################
                    p_list_index += 1
                elif test_method == 'ttest_ind':
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
                    # print('{} 方法，{} 和 {} 之间显著，s = {:.4f}，p = {:.4f}'.format(test_method, labels_name[i1], labels_name[i2], eval('s_' + str(i1) + '_' + str(i2)), eval('t_' + str(i1) + '_' + str(i2))))
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

def plot_correlation_figure(data1, data2, ax=None, stats_method='pearson', dots_color=None, line_color=None, title_name='', title_fontsize=10, title_pad=10, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_major_locator=None, x_max_tick_to_one=False, x_max_tick_to_value=1, x_math_text=True, x_one_decimal_place=False, x_percentage=False, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_major_locator=None, y_max_tick_to_one=False, y_max_tick_to_value=1, y_math_text=True, y_one_decimal_place=False, y_percentage=False, asterisk_fontsize=10, summary=False):
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

def plot_matrix_figure(data, ax=None, row_labels_name=[], col_labels_name=[], cmap='bwr', colorbar_label_name='', colorbar_label_pad=0.1, colorbar_tick_fontsize=10, colorbar_tick_rotation=0, x_rotation=60, row_labels_fontsize=10, col_labels_fontsize=10, colorbar=True,  colorbar_label_fontsize=10, title_name='', title_fontsize=15, title_pad=20, vmax=None, vmin=None, aspect='equal', **kwargs):
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