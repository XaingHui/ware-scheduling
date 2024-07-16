import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.rcsetup import cycler


def analysis_csv(input_csv, output_csv):
    # 从CSV文件读取数据
    df = pd.read_csv(input_csv, header=None,
                     names=["action", "agent_position", "target_position", "total_reward", "elapsed_time",
                            "conflict_count"])

    # 转换 'elapsed_time' 列为数值类型
    df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    df['total_reward'] = pd.to_numeric(df['total_reward'], errors='coerce')
    # 转换 'conflict_count' 列为数值类型
    df['conflict_count'] = pd.to_numeric(df['conflict_count'], errors='coerce')

    # 删除转换后出现 NaN 值的行
    df = df.dropna(subset=['elapsed_time'])

    # 找到 elapsed_time 为负数的索引（表示任务结束）
    end_indices = df[df['elapsed_time'].diff() < 0].index.tolist()

    # 提取每个任务的相关信息
    tasks = []
    for i in range(len(end_indices)):
        start_index = 0 if i == 0 else end_indices[i - 1] + 1
        end_index = int(end_indices[i])  # 将 end_index 转换为整数
        task_data = df.iloc[start_index:end_index - 1]
        total_rows = len(task_data)
        elapsed_time = task_data['elapsed_time'].iloc[-1]  # 取最后一行的值即可
        reward = task_data['total_reward'].iloc[-1]  # 取最后一行的值即可
        total_elapsed_time = task_data['elapsed_time'].sum()
        total_reward = task_data['total_reward'].sum()
        conflict_count = task_data['conflict_count'].iloc[-1]  # 取最后一行的值即可
        tasks.append([total_rows, total_elapsed_time, total_reward, elapsed_time, reward, conflict_count])

    # 创建DataFrame保存结果
    result_df = pd.DataFrame(tasks,
                             columns=['total_rows', 'total_elapsed_time', 'total_reward', 'elapsed_time', 'reward',
                                      'conflict_count'])

    # 将结果保存为CSV文件
    result_df.to_csv(output_csv, index=False)


# Specify a font that supports a wide range of Unicode characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei is a Chinese font, you can choose a different one if needed
plt.rcParams['axes.unicode_minus'] = False  # This is to avoid minus sign display issues in some fonts


# def plot_task_analysis(dataframe):
#     num_colors = len(dataframe) - 60
#     cm = plt.get_cmap('gist_rainbow')
#
#     fig = plt.figure(figsize=(12, 5))
#
#     ax = fig.add_subplot(121)
#     # ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
#     # ax.set_prop_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
#     colors = [cm(1. * i / num_colors) for i in range(num_colors)]
#     ax.set_prop_cycle(cycler('color', colors))
#
#     ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_reward'] / dataframe['total_rows'])[:50],
#             linewidth=2, label='Mean_reward')
#     ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['reward'][:50],
#             linewidth=2, label='total_reward')
#     # for k in ref_discount_rews:
#     #     ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
#     # # ax.plot(max_rew_lr_curve, linewidth=2, label='A2C max')
#
#     plt.legend(loc=4)
#     plt.xlabel("Task", fontsize=20)
#     plt.ylabel("Discounted Total Reward", fontsize=20)
#
#     ax = fig.add_subplot(122)
#     # ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
#     # ax.set_prop_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
#     colors = [cm(1. * i / num_colors) for i in range(num_colors)]
#     ax.set_prop_cycle(cycler('color', colors))
#
#     ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_elapsed_time'] / dataframe['total_rows'])[:50], linewidth=2, label='Mean_time')
#     ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['elapsed_time'][:50], linewidth=2, label='Total_time')
#     # for k in ref_discount_rews:
#     #     ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)
#     #
#     # value = np.concatenate(ref_slow_down['GenAlg'])
#     # ax.plot(value, linewidth=2, label='GenAlg')
#
#     plt.legend(loc=1)
#     plt.xlabel("Task", fontsize=20)
#     plt.ylabel("Elapsed", fontsize=20)
#     plt.savefig("dqn.pdf", format='PDF')
#     plt.show()
#
#
#
# if __name__ == '__main__':
#     analysis_csv('simulation_records.csv', 'task_analysis_result.csv')
#     df = pd.read_csv('task_analysis_result.csv')
#     plot_task_analysis(df)

# def plot_task_analysis(dataframe):
#     num_colors = len(dataframe) - 60
#     cm = plt.get_cmap('gist_rainbow')
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     colors = [cm(1. * i / num_colors) for i in range(num_colors)]
#     ax.set_prop_cycle(cycler('color', colors))
#
#     ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_reward'] / dataframe['total_rows'])[:50],
#             linewidth=2, label='Mean_reward')
#     ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['reward'][:50],
#             linewidth=2, label='total_reward')
#
#     plt.legend(loc=4)
#     plt.xlabel("Task", fontsize=20)
#     plt.ylabel("Discounted Total Reward", fontsize=20)
#     plt.show()
#
# if __name__ == '__main__':
#     analysis_csv('simulation_records.csv', 'task_analysis_result.csv')
#     df = pd.read_csv('task_analysis_result.csv')
#     plot_task_analysis(df)

def plot_elapsed_analysis(dataframe):
    num_colors = len(dataframe) - 60
    cm = plt.get_cmap('gist_rainbow')

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))

    # ax.plot((dataframe.index.astype(int) + 1)[:50], (dataframe['total_elapsed_time'] / dataframe['total_rows'])[:50],
    #         linewidth=2, label='Mean_time')
    # ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['elapsed_time'][:50],
    #         linewidth=2, label='Total_time')
    #
    # plt.legend(loc=1)
    # plt.xlabel("Task", fontsize=20)
    # plt.ylabel("Elapsed", fontsize=20)
    # plt.show()

    ax.plot((dataframe.index.astype(int) + 1)[:50], dataframe['conflict_count'][:50], linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Conflict_count')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框
    plt.savefig("Conflict_Count.pdf", format='pdf')

    # Adjust label font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)

    plt.show()

if __name__ == '__main__':
    analysis_csv('simulation_records.csv', 'task_analysis_result.csv')
    df = pd.read_csv('task_analysis_result.csv')
    plot_elapsed_analysis(df)
