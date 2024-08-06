# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx

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

# 定义起点和终点
start_node = 1
end_node = 16
print("nodes: ", G.nodes())  # 输出所有的节点
print("edges: ", G.edges())  # 输出所有的边
print("number_of_edges: ", G.number_of_edges())  # 边的条数，只有一条边，就是（2，3）

# 使用Dijkstra算法找到最短路径
shortest_path = nx.dijkstra_path(G, start_node, end_node)
shortest_path_length = nx.dijkstra_path_length(G, start_node, end_node)

print("最短路径:", shortest_path)
print("路径长度:", shortest_path_length)

# 绘制图形
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
plt.figure(figsize=(20, 15))

# 绘制节点和边
nx.draw(G, positions, with_labels=False, node_color='skyblue', node_size=1500, font_size=10,
        font_color='black', arrows=True)
nx.draw_networkx_labels(G, positions, font_size=12)

# 高亮最短路径
path_edges = list(zip(shortest_path, shortest_path[1:]))
nx.draw_networkx_edges(G, positions, edgelist=path_edges, edge_color='r', width=2)

# 显示图形
plt.title("Directed Graph with Shortest Path Highlighted")
plt.show()
