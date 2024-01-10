import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import statsmodels.api as sm


def plot_one_group_bar_figure(data, test_method, labels_name=None, ax=None, width=0.5, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, asterisk_fontsize=10, multicorrect=False, **kwargs):
    """
    ## 说明:
    该函数用于绘制一个单组带点bar图。

    ## 参数:
    - data: list，每个元素是一个(n,)的数组。
    - labels_name: list，用于展示在图上的x轴的labels的名字。
    - ax: ax，事先创建的Axes。
    - width: float，bar的宽度。
    - colors: list，颜色列表。
    - title_name: str，标题。
    - title_fontsize: float，标题字体大小。
    - title_pad: float，图像标题与图像间隔。
    - x_label_name: str，x轴的标题。
    - x_label_fontsize: float，x轴标题的字体大小。
    - x_tick_fontsize: float，x轴上labels的字体大小。
    - x_tick_rotation: float，x轴上labels旋转角度。
    - x_label_ha: str，“left/center/right”，默认“center”，如果需要旋转，建议选“right”。
    - y_label_name: str，y轴的标题。
    - y_label_fontsize: float，y轴标题的字体大小。
    - y_tick_fontsize: float，y轴上labels的字体大小。
    - y_tick_rotation: float，y轴上labels旋转角度。
    - y_max_tick_to_one: bool，默认“False”，开启后y轴可以超过一个值，但是tick最多只显示到该值，配合“y_max_tick_to_value”使用。
    - y_max_tick_to_value: float，设置后y轴可以超过一个值，但是tick最多只显示到该值，需要打开“y_max_tick_to_one”。
    - math_text: bool，默认“True”，设置y轴科学计数法。
    - one_decimal_place: bool，默认“False”，设置y轴tick保留1位小数，会与“math_text”冲突。
    - percentage: bool，默认“False”，设置y轴为百分数的显示效果，会与“math_text”冲突。
    - ax_min_is_0: bool，默认“False”，设置y轴最小值为0。
    - asterisk_fontsize: float，星号的字体大小。

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    输入最多参数调用:
    >>> im1 = rjx_plot_matrix_figure(data, labels, labels, ax=ax1, cmap='Reds', cbarlabel='Title of the colorbar', title='This is a title!')
    输入最少参数调用:
    >>> im2 = rjx_plot_matrix_figure(data, ax=ax2)
    """
    # # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    if labels_name is None:
        labels_name = [str(i) for i in range(len(data))]
    if colors is None:
        colors = ['k'] * len(data)
    ##################################################################################################
    bar_mean_list, bar_SD_list, bar_SE_list = [], [], []
    for i in range(0,len(labels_name)):
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
    ax.bar(x,bar_mean_list,width=width,color=colors,alpha=0.5)
    ax.errorbar(x, bar_mean_list, bar_SE_list, fmt='none', linewidth=1, capsize=3, color='black')
    for i in range(0,len(x)):
        ax.scatter(eval('x'+str(i)+'_dots'), eval('data'+str(i)), c=colors[i], s=35, edgecolors='white', linewidths=1, alpha=0.9)
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
        if np.min(data[i]) < 0.01 or np.max(data[i]) > 100:  # y轴设置科学计数法
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
    # 如果y轴最小值不需要设置成0，则设置为黄金比例
    if ax_min_is_0:
        ax.set_ylim(0, ax_max)
    else:
        ax.set_ylim(ax_min, ax_max)
    ############################################## 标星号 ############################################
    t_count = 0
    for i1 in range(len(labels_name)):
        for i2 in range(i1+1, len(labels_name)):
            if test_method == 'ttest_ind':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_ind(data[i1], data[i2])
            elif test_method == 'ttest_rel':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_rel(data[i1], data[i2])
            elif test_method == 'mannwhitneyu':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.mannwhitneyu(data[i1], data[i2], alternative='two-sided')
            if multicorrect == True:
                globals()['t_' + str(i1) + '_' + str(i2)] = eval('t_' + str(i1) + '_' + str(i2)) * (len(labels_name) * (len(labels_name) - 1)) / 2  # 多重比较校正，直接将p值乘以比较次数bonferroni校正
            if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                t_count +=1
                print('{} 方法，{} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i1], labels_name[i2], round(eval('s_' + str(i1) + '_' + str(i2)), 3), round(eval('t_' + str(i1) + '_' + str(i2)), 3)))
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
    return

def plot_multi_group_bar_figure(data, test_method, legend_name=None, labels_name=None, ax=None, width=0.2, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=30, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, math_text=True, y_max_tick_to_one=False, y_max_tick_to_value=1, one_decimal_place=False, percentage=False, legend_fontsize=10, legend_location='best', legend_bbox_location=None, ax_min_is_0=False, asterisk_size=10, multicorrect=False, **kwargs):
    """
    ## 说明:
    该函数用于绘制一个多组的bar图。

    ## 参数:
    - data: list，每个元素是一个(n,m)的数组，n为采样数量，m为组数。
    - legend_name: list，用于展示在图上的legend的名字。
    - labels_name: list，用于展示在图上的x轴的labels的名字。
    - ax: ax，事先创建的Axes。
    - width: float，bar的宽度。
    - colors: list，颜色列表。
    - title_name: str，标题。
    - title_fontsize: float，标题字体大小。
    - title_pad: float，图像标题与图像间隔。
    - x_label_name: str，x轴的标题。
    - x_label_fontsize: float，x轴标题的字体大小。
    - x_tick_fontsize: float，x轴上labels的字体大小。
    - x_tick_rotation: float，x轴上labels旋转角度。
    - y_label_name: str，y轴的标题。
    - y_label_fontsize: float，y轴标题的字体大小。
    - y_tick_fontsize: float，y轴上labels的字体大小。
    - y_tick_rotation: float，y轴上labels旋转角度。
    - math_text: bool，默认“True”，设置y轴科学计数法。
    - y_max_tick_to_one: bool，默认“False”，开启后y轴可以超过一个值，但是tick最多只显示到该值，配合“y_max_tick_to_value”使用。
    - y_max_tick_to_value: float，设置后y轴可以超过一个值，但是tick最多只显示到该值，需要开启“y_max_tick_to_one”。
    - one_decimal_place: bool，默认“False”，设置y轴tick保留1位小数，会与“math_text”冲突。
    - percentage: bool，默认“False”，设置y轴为百分数的显示效果，会与“math_text”冲突。
    - legend_fontsize: float，legend的字体大小。
    - legend_location: “upper left, upper right, lower left, lower right, best”，默认“best”。
    - legend_bbox_location: tuple，指定legend位置。开启后“legend_location”失效。
    - ax_min_is_0: bool，默认“False”，设置y轴最小值为0。
    - asterisk_size: float，星号的字体大小。
    - test_method: str，检验方法，默认为'ttest_ind'。

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    >>> rjx_plot_multi_group_bar_figure(legend_name, colors=colors, labels_name=labels_name, legend_location='lower left')
    """
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
                if multicorrect==True:
                    globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) * len(labels_name) * (len(legend_name) * (len(legend_name) - 1)) / 2   # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    print('{} 方法，{} 组里的 {} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i0], legend_name[i1], legend_name[i2], round(eval('s' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3), round(eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3)))
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

def plot_matrix_figure(data, row_labels_name=[], col_labels_name=[], ax=None, cmap='bwr', cbarlabel_name='',cbartick_fontsize=10, cbartick_rotation=0, x_rotation=60, row_labels_fontsize=5, col_labels_fontsize=5, cbarlabel_fontsize=10, title_name='', title_fontsize=15, title_pad=20, vmax=None, vmin=None, **kwargs):
    """
    ## 说明:
    该函数用于绘制一个矩阵图。

    ## 参数:
    - data: array，矩阵数据。
    - row_labels_name: list，矩阵y轴label。
    - col_labels_name: list，矩阵x轴label。
    - ax: ax，事先创建的Axes。
    - cmap: str，colorbar颜色。
    - cbarlabel_name: str，colorbar的label。
    - x_rotation: float，x轴label旋转的角度。
    - row_labels_fontsize: float，y轴label的字体大小。
    - col_labels_fontsize: float，x轴label的字体大小。
    - cbarlabel_fontsize: float，colorbar的label的字体大小。
    - title_name: str，图像标题。
    - tilte_fontsize: float，图像标题的字体大小。
    - title_pad: float，图像标题与图像间隔。
    - vmax: float，设置colorbar的最大值
    - vmin: float，设置colorbar的最小值

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    输入最多参数调用:
    >>> plot_matrix_figure(data, labels, labels, ax=ax1, cmap='Reds', cbarlabel_name='Title of the colorbar', title_name='This is a title!', vmin=0, vmax=0.1)
    输入最少参数调用:
    >>> plot_matrix_figure(data, ax=ax2)
    """
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    # 指定最大最小值
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    # 画图
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title_name, fontsize=title_fontsize, pad=title_pad)
    # 根据矩阵文件读取获得x、y轴长度以及标上label，并定义xy的labels字体大小
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels_name)
    ax.set_xticklabels(col_labels_name, fontsize=col_labels_fontsize)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels_name)
    ax.set_yticklabels(row_labels_name, fontsize=row_labels_fontsize)
    # 获取当前图形的坐标轴
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # pad 控制 colorbar 与图的距离
    # 创建colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(cbarlabel_name, rotation=-90, va="bottom", fontsize=cbarlabel_fontsize)
    cbar.ax.tick_params(axis='y', which='major', labelsize=cbartick_fontsize, rotation=cbartick_rotation)
    # 调整 colorbar 的高度，使其与图的高度相同
    ax_height = ax.get_position().height
    cax.set_position([cax.get_position().x0, cax.get_position().y0, cax.get_position().width, ax_height])
    # 旋转x轴label并对齐到每个cell中线
    plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha="right", rotation_mode="anchor")  # 其中“right”表示label最右边字母对齐到每个cell的中线
    return

def plot_one_group_violin_figure(data, test_method, labels_name=None, ax=None, width=0.5, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_label_ha='center', y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_max_tick_to_one=False, y_max_tick_to_value=1, math_text=True, one_decimal_place=False, percentage=False, ax_min_is_0=False, asterisk_fontsize=10, multicorrect=False, **kwargs):
    """
    ## 说明:
    该函数用于绘制一个单组小提琴图图。

    ## 参数:
    - data: list，每个元素是一个(n,)的数组。
    - labels_name: list，用于展示在图上的x轴的labels的名字。
    - ax: ax，事先创建的Axes。
    - width: float，bar的宽度。
    - colors: list，颜色列表。
    - title_name: str，标题。
    - title_fontsize: float，标题字体大小。
    - title_pad: float，图像标题与图像间隔。
    - x_label_name: str，x轴的标题。
    - x_label_fontsize: float，x轴标题的字体大小。
    - x_tick_fontsize: float，x轴上labels的字体大小。
    - x_tick_rotation: float，x轴上labels旋转角度。
    - x_label_ha: str，“left/center/right”，默认“center”，如果需要旋转，建议选“right”。
    - y_label_name: str，y轴的标题。
    - y_label_fontsize: float，y轴标题的字体大小。
    - y_tick_fontsize: float，y轴上labels的字体大小。
    - y_tick_rotation: float，y轴上labels旋转角度。
    - y_max_tick_to_one: bool，默认“False”，开启后y轴可以超过一个值，但是tick最多只显示到该值，配合“y_max_tick_to_value”使用。
    - y_max_tick_to_value: float，设置后y轴可以超过一个值，但是tick最多只显示到该值，需要打开“y_max_tick_to_one”。
    - math_text: bool，默认“True”，设置y轴科学计数法。
    - one_decimal_place: bool，默认“False”，设置y轴tick保留1位小数，会与“math_text”冲突。
    - percentage: bool，默认“False”，设置y轴为百分数的显示效果，会与“math_text”冲突。
    - ax_min_is_0: bool，默认“False”，设置y轴最小值为0。
    - asterisk_fontsize: float，星号的字体大小。

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    >>> rjx_plot_one_group_violin_figure(data, labels_name, ax=ax2, colors=colors, math_text=False, one_decimal_place=True, x_tick_rotation=30, x_label_ha='right')
    """
    # # 设置部分默认值
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
        pc.set_alpha(0.5)
    violin_parts['cmaxes'].set_edgecolor('black')
    violin_parts['cmaxes'].set_alpha(0.5)
    violin_parts['cmins'].set_edgecolor('black')
    violin_parts['cmins'].set_alpha(0.5)
    violin_parts['cbars'].set_edgecolor('black')
    violin_parts['cbars'].set_alpha(0.5)
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
    # y轴设置科学计数法
    if math_text:
        if np.min(data[i]) < 0.01 or np.max(data[i]) > 100:  # y轴设置科学计数法
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
        all_data_ymax.append(np.max(data[i]))
        all_data_ymin.append(np.min(data[i]))
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
    ############################################## 标星号 ############################################
    t_count = 0
    for i1 in range(len(labels_name)):
        for i2 in range(i1+1, len(labels_name)):
            if test_method == 'ttest_ind':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_ind(data[i1], data[i2])
            elif test_method == 'ttest_rel':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.ttest_rel(data[i1], data[i2])
            elif test_method == 'mannwhitneyu':
                globals()['s_' + str(i1) + '_' + str(i2)], globals()['t_' + str(i1) + '_' + str(i2)] = stats.mannwhitneyu(data[i1], data[i2], alternative='two-sided')
            if multicorrect == True:
                globals()['t_' + str(i1) + '_' + str(i2)] = eval('t_' + str(i1) + '_' + str(i2)) * (len(labels_name) * (len(labels_name) - 1)) / 2  # 多重比较校正，直接将p值乘以比较次数bonferroni校正
            if eval('t_' + str(i1) + '_' + str(i2)) <= 0.05:
                t_count +=1
                print('{} 方法，{} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i1], labels_name[i2], round(eval('s_' + str(i1) + '_' + str(i2)), 3), round(eval('t_' + str(i1) + '_' + str(i2)), 3)))
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
    return

def plot_multi_group_violin_figure(data, test_method, legend_name=None, labels_name=None, ax=None, width=0.2, colors=None, title_name='', title_fontsize=15, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=30, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, math_text=True, y_max_tick_to_one=False, y_max_tick_to_value=1, one_decimal_place=False, percentage=False, legend_fontsize=10, legend_location='best', legend_bbox_location=None, ax_min_is_0=False, asterisk_size=10, multicorrect=False, **kwargs):
    """
    ## 说明:
    该函数用于绘制一个多组的bar图。

    ## 参数:
    - data: list，每个元素是一个(n,m)的数组，n为采样数量，m为组数。
    - legend_name: list，用于展示在图上的legend的名字。
    - labels_name: list，用于展示在图上的x轴的labels的名字。
    - ax: ax，事先创建的Axes。
    - width: float，bar的宽度。
    - colors: list，颜色列表。
    - title_name: str，标题。
    - title_fontsize: float，标题字体大小。
    - title_pad: float，图像标题与图像间隔。
    - x_label_name: str，x轴的标题。
    - x_label_fontsize: float，x轴标题的字体大小。
    - x_tick_fontsize: float，x轴上labels的字体大小。
    - x_tick_rotation: float，x轴上labels旋转角度。
    - y_label_name: str，y轴的标题。
    - y_label_fontsize: float，y轴标题的字体大小。
    - y_tick_fontsize: float，y轴上labels的字体大小。
    - y_tick_rotation: float，y轴上labels旋转角度。
    - math_text: bool，默认“True”，设置y轴科学计数法。
    - y_max_tick_to_one: bool，默认“False”，开启后y轴可以超过一个值，但是tick最多只显示到该值，配合“y_max_tick_to_value”使用。
    - y_max_tick_to_value: float，设置后y轴可以超过一个值，但是tick最多只显示到该值，需要开启“y_max_tick_to_one”。
    - one_decimal_place: bool，默认“False”，设置y轴tick保留1位小数，会与“math_text”冲突。
    - percentage: bool，默认“False”，设置y轴为百分数的显示效果，会与“math_text”冲突。
    - legend_fontsize: float，legend的字体大小。
    - legend_location: “upper left, upper right, lower left, lower right, best”，默认“best”。
    - legend_bbox_location: tuple，指定legend位置。开启后“legend_location”失效。
    - ax_min_is_0: bool，默认“False”，设置y轴最小值为0。
    - asterisk_size: float，星号的字体大小。

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    >>> rjx_plot_multi_group_bar_figure(legend_name, colors=colors, labels_name=labels_name, legend_location='lower left')
    """
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
    x = np.arange(len(legend_name))
    position = []
    for i in range(0, len(legend_name)):
        violin_data = data[i]
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
        for i1 in range(data[i].shape[1]):
            each_group_data = data[i][:, i1]
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
        if np.min(data[i]) < 0.01 or np.max(data[i]) > 100:  # y轴设置科学计数法
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
        all_data_ymax.append(np.max(data[i]))
        all_data_ymin.append(np.min(data[i]))
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
                    stats.ttest_ind(data[i1][:,i0], data[i2][:,i0])
                elif test_method == 'ttest_rel':
                    globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                    stats.ttest_rel(data[i1][:,i0], data[i2][:,i0])
                elif test_method == 'mannwhitneyu':
                    globals()['s' + str(i0) + '_' + str(i1) + '_' + str(i2)], globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = \
                    stats.mannwhitneyu(data[i1][:,i0], data[i2][:,i0], alternative='two-sided')
                if multicorrect == True:
                    globals()['t' + str(i0) + '_' + str(i1) + '_' + str(i2)] = eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) * len(labels_name) * (len(legend_name) * (len(legend_name) - 1)) / 2   # 多重比较校正，直接将p值乘以比较次数bonferroni校正
                if eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)) <= 0.05:
                    t_count +=1
                    print('{} 方法，{} 组里的 {} 和 {} 之间显著，s = {}，p = {}'.format(test_method, labels_name[i0], legend_name[i1], legend_name[i2], round(eval('s' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3), round(eval('t' + str(i0) + '_' + str(i1) + '_' + str(i2)), 3)))
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

def plot_correlation_figure(data, ax=None, dots_color=None, line_color=None, title_name='', title_fontsize=12, title_pad=20, x_label_name='', x_label_fontsize=10, x_tick_fontsize=10, x_tick_rotation=0, x_major_locator=None, x_max_tick_to_one=False, x_max_tick_to_value=1, x_math_text=True, x_one_decimal_place=False, x_percentage=False, y_label_name='', y_label_fontsize=10, y_tick_fontsize=10, y_tick_rotation=0, y_major_locator=None, y_max_tick_to_one=False, y_max_tick_to_value=1, y_math_text=True, y_one_decimal_place=False, y_percentage=False, asterisk_fontsize=10):
    """
    ## 说明:
    该函数用于绘制一个相关点线图。

    ## 参数:
    - data: arr，是一个(n, 2)的数组，n为采样数量。
    - ax: ax，事先创建的Axes。
    - dots_color: str，点的颜色。
    - line_color: str，线的颜色。
    - title_name: str，标题。
    - title_fontsize: float，标题字体大小。
    - title_pad: float，图像标题与图像间隔。
    - x_label_name: str，x轴的标题。
    - x_label_fontsize: float，x轴标题的字体大小。
    - x_tick_fontsize: float，x轴上labels的字体大小。
    - x_tick_rotation: float，x轴上labels旋转角度。
    - x_major_locator: float，x轴上ticks之间的间隔。
    - x_max_tick_to_one: bool，默认“False”，开启后x轴可以超过一个值，但是tick最多只显示到该值，配合“x_max_tick_to_value”使用。
    - x_max_tick_to_value: float，设置后x轴可以超过一个值，但是tick最多只显示到该值，需要开启“x_max_tick_to_one”。
    - x_math_text: bool，默认“True”，设置x轴科学计数法。
    - x_one_decimal_place: bool，默认“False”，设置x轴tick保留1位小数，会与“x_math_text”冲突。
    - x_percentage: bool，默认“False”，设置x轴为百分数的显示效果，会与“x_math_text”冲突。
    - y_label_name: str，y轴的标题。
    - y_label_fontsize: float，y轴标题的字体大小。
    - y_tick_fontsize: float，y轴上labels的字体大小。
    - y_tick_rotation: float，y轴上labels旋转角度。
    - y_major_locator: float，y轴上ticks之间的间隔。
    - y_max_tick_to_one: bool，默认“False”，开启后y轴可以超过一个值，但是tick最多只显示到该值，配合“y_max_tick_to_value”使用。
    - y_max_tick_to_value: float，设置后y轴可以超过一个值，但是tick最多只显示到该值，需要开启“y_max_tick_to_one”。
    - y_math_text: bool，默认“True”，设置y轴科学计数法。
    - y_one_decimal_place: bool，默认“False”，设置y轴tick保留1位小数，会与“y_math_text”冲突。
    - y_percentage: bool，默认“False”，设置x轴为百分数的显示效果，会与“y_math_text”冲突。
    - asterisk_fontsize: float，星号的字体大小。

    ## 返回值:
    无。

    ## 注意事项:
    暂无！y(=~ω~=)y

    ## 例子:
    >>> plot_correlation_figure(data, line_color='k')
    """
    # 设置部分默认值
    if ax is None:
        ax = plt.gca()
    ##################################################################################################
    data1, data2 = data[:, 0], data[:, 1]
    exog, endog = sm.add_constant(data1), data2
    model = sm.OLS(endog, exog, missing="drop")
    results = model.fit()
    print(results.summary())  # 第2种展示结果的方式
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
    s, p = stats.spearmanr(data1, data2)
    print('斯皮尔曼相关性，r={}，p={}'.format(round(s, 3), round(p, 3)))
    s, p = stats.pearsonr(data1, data2)
    print('皮尔逊相关性，r={}，p={}'.format(round(s, 3), round(p, 3)))
    x_start, x_end = ax.get_xlim()
    x_width = x_end - x_start
    y_start, y_end = ax.get_ylim()
    y_width = y_end - y_start
    ax.text(x_start + x_width/ 10, y_start + 9 * y_width / 10, 'r = '+str(round(s, 3)), va='center', fontsize=asterisk_fontsize)  # 参数保留小数点后3位
    if 0.01 < p < 0.05:
        ax.text(x_start + x_width / 10, y_start + 8 * y_width / 10, '*', va='center', fontsize=asterisk_fontsize)
    elif 0.001 < p < 0.01:
        ax.text(x_start + x_width / 10, y_start + 8 * y_width / 10, '**', va='center', fontsize=asterisk_fontsize)
    elif p < 0.001:
        ax.text(x_start + x_width / 10, y_start + 8 * y_width / 10, '***', va='center', fontsize=asterisk_fontsize)
    else:
        ax.text(x_start + x_width / 10, y_start + 8 * y_width / 10, 'p = '+str(round(p,3)), va='center', fontsize=asterisk_fontsize)  # 参数保留小数点后3位
    return
