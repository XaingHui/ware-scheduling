# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# 设置 matplotlib 使用 SimHei 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 创建一个有向图
G = nx.Graph()

# 添加边（假设每条边的权重为1）
edges = [
    (1, 2), (1, 3),
    (2, 4), (2, '预处理堆场'), (2, '综合堆场'),
    (3, 4), (3, '总组堆场'), (3, '车场'),
    (4, 5), (4, 6), (4, '小组成立车间'), (4, '曲面车间'), (4, '11号平台'), (4, '平直中心'),
    (5, 7), (5, '10号平台'),
    (6, 7), (6, 11), (6, '平直广场'), (6, '涂装车间'),
    (7, 12),
    (8, 9), (8, 13), (8, '舾装车间'),
    (9, 10), (9, 14), (9, '8号平台'),
    (10, 15), (10, 11), (10, '9号平台'),
    (11, 16), (11, '涂装广场'),
    (12, '涂装堆场'),
    (13, 14), (13, '舾装堆场'),
    (14, 15),
    (15, 16), (15, '9B平台')
]

# 将边添加到图中，权重为1
for edge in edges:
    G.add_edge(*edge, weight=1)

# 读取 Excel 文件中的数据
input_file = 'input.xls'  # 输入文件路径
output_file = 'output.xlsx'  # 输出文件路径

try:
    data = pd.read_excel(input_file, engine='xlrd')  # 指定引擎为 'xlrd'
except Exception as e:
    print(f"读取 Excel 文件时出错: {e}")
    exit()

# 创建一个 DataFrame 来存储结果
results = pd.DataFrame(columns=['起点', '终点', '最短路径', '路径长度'])

# 定义节点的位置（x, y）
positions = {
    1: (0, 0), 2: (0, 28), 3: (7, 0), 4: (7, 28), 5: (7, 40), 6: (13, 28), 7: (13, 40),
    8: (26, 0), 9: (26, 14), 10: (26, 22), 11: (26, 28), 12: (26, 40), 13: (50, 0), 14: (50, 14),
    15: (50, 22), 16: (50, 28),
    '车场': (8, 3), '8号平台': (28, 15), '平直中心': (8, 29), '曲面车间': (5, 26),
    '涂装车间': (14, 29), '小组成立车间': (4, 27), '11号平台': (6, 29), '10号平台': (8, 39),
    '9B平台': (48, 24), '9号平台': (28, 20), '舾装车间': (27, 5), '舾装堆场': (46, 5),
    '平直广场': (12, 29), '涂装广场': (24, 30), '综合堆场': (1, 29), '总组堆场': (6, 10),
    '预处理堆场': (1, 26), '涂装堆场': (27, 38), '总组平台': (23, 24)
}

# 处理每一行，计算最短路径并保存图像
for index, row in data.iterrows():
    start_node = row['起点']
    end_node = row['终点']

    try:
        # 使用Dijkstra算法找到最短路径
        shortest_path = nx.dijkstra_path(G, start_node, end_node)
        shortest_path_length = nx.dijkstra_path_length(G, start_node, end_node)

        # 存储结果
        new_row = pd.DataFrame({
            '起点': [start_node],
            '终点': [end_node],
            '最短路径': [str(shortest_path)],
            '路径长度': [shortest_path_length]
        })
        results = pd.concat([results, new_row], ignore_index=True)

        # 绘制图形
        plt.figure(figsize=(20, 15))
        nx.draw(G, positions, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, font_color='black',
                arrows=True)
        nx.draw_networkx_labels(G, positions, font_size=12)

        # 高亮最短路径
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=path_edges, edge_color='r', width=2)

        # 显示图形
        plt.title(f"最短路径: {start_node} -> {end_node}")
        plt.savefig(f"shortest_path_{start_node}_to_{end_node}.png")
        plt.close()

    except nx.NetworkXNoPath:
        print(f"节点 {start_node} 和 {end_node} 之间没有路径")

# 将结果写入 Excel 文件
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    data.to_excel(writer, sheet_name='输入数据', index=False)
    results.to_excel(writer, sheet_name='最短路径', index=False)
