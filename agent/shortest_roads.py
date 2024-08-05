# -*- coding: utf-8 -*-
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加边（假设每条边的权重为1）
edges = [
    (1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 9), (4, 6), (4, 7), (5, 10), (6, 11),
    (7, 12), (8, 9), (9, 10), (9, 14), (10, 15), (11, 16), (13, 9), (14, 15),
    (9, 8), (15, 10), (12, 7)
]

# 将边添加到图中，权重为1
for edge in edges:
    G.add_edge(*edge, weight=1)

# 定义起点和终点
start_node = 1
end_node= 16

# 使用Dijkstra算法找到最短路径
shortest_path = nx.dijkstra_path(G, start_node, end_node)
shortest_path_length_new = nx.dijkstra_path_length(G, start_node, end_node)

print("最短路径:", shortest_path)
print("路径长度:", shortest_path_length)
