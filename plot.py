import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.rcsetup import cycler


def merge_csv(input_csv, input_csv_random):
    # 读取两个 CSV 文件
    df1 = pd.read_csv(input_csv)
    df2 = pd.read_csv(input_csv_random)

    # 给第二个 CSV 文件的列名加上 "random" 后缀
    df2.columns = [col + '_random' for col in df2.columns]

    # 合并两个 DataFrame
    merged_df = pd.concat([df1, df2], axis=1)

    # 保存合并后的 DataFrame 到新的 CSV 文件
    merged_df.to_csv('task_analysis.csv', index=False)


# Specify a font that supports a wide range of Unicode characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei is a Chinese font, you can choose a different one if needed
plt.rcParams['axes.unicode_minus'] = False  # This is to avoid minus sign display issues in some fonts


def plot_task_analysis(dataframe):
    num_colors = len(dataframe) - 60
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    # ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    # ax.set_prop_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('blue', colors))

    ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_reward'] / dataframe['total_rows'])[:50],
            linewidth=2, label='Mean1_reward')
    ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['reward'][:50],
            linewidth=2, label='Article_reward')
    # 计算 'conflict_count_random' 的平均值
    average_random = dataframe['reward_random'].mean()

    # 将 'conflict_count_random' 的平均值延长到 'total_rows'
    random_extension = np.full(dataframe['total_rows'].max(), average_random)

    # 将 'conflict_count_random' 的平均值赋值给 'conflict_count_random' 列
    dataframe['reward_random'] = random_extension[:len(dataframe)]
    ax.plot(dataframe.index.astype(int) + 1, dataframe['reward_random'] / dataframe['total_rows'],
            linewidth=2, label='Random')
    ax.plot(dataframe.index.astype(int) + 1,dataframe['reward_random'] , linewidth=2, label='Mean2_reward')




    # for k in ref_discount_rews:
    #     ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    # # ax.plot(max_rew_lr_curve, linewidth=2, label='A2C max')

    plt.legend(loc=4)
    plt.xlabel("Task", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    # ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    # ax.set_prop_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))

    ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_elapsed_time'] / dataframe['total_rows'])[:50], linewidth=2,
            label='Mean1_time')
    ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['elapsed_time'][:50], linewidth=2, label='Article_time')
    # 计算 'conflict_count_random' 的平均值
    average_random = dataframe['elapsed_time_random'].mean()

    # 将 'conflict_count_random' 的平均值延长到 'total_rows'
    random_extension = np.full(dataframe['total_rows'].max(), average_random)

    # 将 'conflict_count_random' 的平均值赋值给 'conflict_count_random' 列
    dataframe['elapsed_time_random'] = random_extension[:len(dataframe)]
    ax.plot(dataframe.index.astype(int) + 1, dataframe['elapsed_time_random'] / dataframe['total_rows'],
            linewidth=2, label='Random')
    ax.plot(dataframe.index.astype(int) + 1, dataframe['elapsed_time_random'], linewidth=2, label='Mean2_time')

    # for k in ref_discount_rews:
    #     ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)
    #
    # value = np.concatenate(ref_slow_down['GenAlg'])
    # ax.plot(value, linewidth=2, label='GenAlg')

    plt.legend(loc=1)
    plt.xlabel("Task", fontsize=20)
    plt.ylabel("Elapsed", fontsize=20)

    plt.savefig('compare.pdf', format='PDF')

    # 1. 利用彩虹色绘制总冲突
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))

    # ax.plot(dataframe.index.astype(int) + 1, dataframe['conflict_count'] / dataframe['total_rows'], linewidth=2,
    #         label='Mean')
    ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['conflict_count'][:50], linewidth=2, label='Article')
    # 计算 'conflict_count_random' 的平均值
    # average_random = dataframe['conflict_count_random'].mean()
    #
    # 将 'conflict_count_random' 的平均值延长到 'total_rows'
    # random_extension = np.full(dataframe['total_rows'].max(), dataframe['conflict_count_random'])
    #
    # 将 'conflict_count_random' 的平均值赋值给 'conflict_count_random' 列
    # dataframe['conflict_count_random'] = random_extension[:len(dataframe)]
    ax.plot(dataframe.index.astype(int) + 1, dataframe['conflict_count_random'],
            linewidth=2, label='Random')
    # for k in ref_discount_rews:
    #     ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)
    #
    # value = np.concatenate(ref_slow_down['GenAlg'])
    # ax.plot(value, linewidth=2, label='GenAlg')

    plt.legend(loc=1)
    plt.xlabel("Task", fontsize=20)
    plt.ylabel("Conflict", fontsize=20)


    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(dataframe.index.astype(int) + 1, dataframe['total_conflict_count'] / dataframe['total_rows'], marker='o',
    #         linestyle='-',
    #         color=colors[1], label='conflict_count_real', linewidth=2)
    # ax.plot(dataframe.index.astype(int) + 1, dataframe['total_conflict_count_random'] / dataframe['total_rows_random'],
    #         marker='o', linestyle='-', color=colors[2],
    #         label='conflict_count_random', linewidth=2)
    # ax.set_xlabel('Task')
    # ax.set_ylabel('Mean_conflict_count')
    # ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框


    # Adjust label font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    plt.savefig("compare_conflict.pdf", format='PDF')
    plt.show()


if __name__ == '__main__':
    merge_csv('./agent/task_analysis_result.csv', './agent_random_5/task_analysis_result_random.csv')
    df = pd.read_csv('task_analysis.csv')
    plot_task_analysis(df)


