import copy
import csv
import time
from random import choice

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datetime import datetime, timedelta


class Item:
    def __init__(self, item_id, x, y, length, width, start_time, processing_time, exit_time, time_remain, color):
        self.item_id = item_id
        self.length = length  # 分段长
        self.width = width  # 分段宽
        self.start_time = start_time  # 最早开工时间
        self.processing_time = processing_time  # 加工周期
        self.exit_time = exit_time  # 最早出场时间
        self.x = x  # 物品的 x 坐标
        self.y = y  # 物品的 y 坐标
        self.time_remain = time_remain  # 时间余量
        self.color = color  # 可选：为物品添加颜色属性

    def __str__(self):
        return f"Item ID: {self.item_id}, Size: ({self.length}m x {self.width}m), Bound:({self.x + self.width},{self.y + self.length})" \
               f"Start Time: {self.start_time}, Processing Time: {self.processing_time} days, " \
               f"Exit Time: {self.exit_time}"

    # 可以根据需要添加其他方法，例如移动物品或检查物品与其他物品的冲突
    def get_rectangle(self):
        """
        获取物品的矩形形状。

        返回：
        - (left, top, right, bottom) 元组表示矩形的左上角和右下角坐标
        """
        left = self.x
        top = self.y
        right = self.x + self.width
        bottom = self.y + self.length
        return left, top, right, bottom


class WarehouseEnvironment:
    def __init__(self, width, height, number, time='2017/9/1'):
        self.width = width
        self.height = height
        self.number = number
        self.bias = 0
        self.segment_heights = []  # 存储已添加物品的分段高度
        self.y_positions = []

        self.seg_high = 0  # [seg_mid,seg_high]
        self.seg_mid = 0  # [seg_low,seg_mid]
        self.seg_low = 0  # [0,seg_low]
        self.seg_high_length = 50
        self.seg_mid_length = 52
        self.seg_low_length = 50

        self.segment_high = [20]
        self.segment_mid = []
        self.segment_low = []

        self.grid = np.zeros((height, width), dtype=object)
        self.agent = Item('agent', 0, 0, 5, 5, time, 0, time, 0, 'black')

        self.items = {}
        self.colors = list(mcolors.TABLEAU_COLORS)
        # self.road = {'x': width, 'width': 20, 'color': 'lightgray'}  # 道路属性
        self.roads = []
        # {'x_left': width, 'width': 20, 'color': 'lightblue'},
        # {'x_top': width, 'height': 20, 'color': 'lightpink'},
        # {'x_bottom': width, 'height': 20, 'color': 'lightyellow'}]  # 道路属性
        self.target_position = (0, 0)  # 目标位置
        self.initial_position = (0, 0)

        self.tag_right = False
        self.tag_left = False
        self.tag_bottom = False
        self.tag_top = False

        self.current_time = datetime.strptime(time, '%Y/%m/%d')  # 最早出场时间
        self.agent_has_item = False
        self.total_reward = 0
        self.total_step_time = 0
        self.item = Item('agent', self.agent.x, self.agent.y, 5, 5, time, 0, time, 0, 'black')
        self.task_positions = []
        self.conflict_count = 0
        self.out_list = []
        self.item_random = None
        self.initial_state = {
            'agent_has_item': self.agent_has_item,
            'agent_position': (self.agent.x, self.agent.y),
            'target_positions': self.target_position,
        }
        self.cache_items = []
        self.step_records = []
        self.interfering_items = []
        self.start_time = datetime.now()

    def change_bias(self):
        self.bias = 20

    def choose_road(self, x, y):
        distances = {
            'distance_x_left': x,
            'distance_x_right': self.width - x,
            'distance_y_top': y,
            'distance_y_bottom': self.height - y,
        }

        valid_roads = []

        if self.tag_left:
            valid_roads.append('distance_x_left')
        if self.tag_right:
            valid_roads.append('distance_x_right')
        if self.tag_top:
            valid_roads.append('distance_y_top')
        if self.tag_bottom:
            valid_roads.append('distance_y_bottom')

        if valid_roads:
            min_distance_road = min(valid_roads, key=lambda road: distances[road])
            print(f"Choosing road: {min_distance_road}")
            if min_distance_road == 'distance_x_left':
                self.task_positions.append((0, self.agent.y))

            elif min_distance_road == 'distance_x_right':
                self.task_positions.append((self.width, self.agent.y))

            elif min_distance_road == 'distance_y_top':
                self.task_positions.append((self.agent.x, 0))

            elif min_distance_road == 'distance_y_bottom':
                self.task_positions.append((self.agent.x, self.height - self.agent.length))

        else:
            print("No valid roads found.")

        print("Task positions:", self.task_positions)

    def setRoads(self, *names):
        roads = [{'name': 'right', 'width': 20, 'color': 'lightgray'},
                 {'name': 'left', 'width': 20, 'color': 'lightblue'},
                 {'name': 'top', 'height': 20, 'color': 'lightpink'},
                 {'name': 'bottom', 'height': 20, 'color': 'lightyellow'}]  # 道路属性

        for name in names:
            if name == 'right':
                self.roads.append(roads[0])
                self.tag_right = True
            if name == 'left':
                self.roads.append(roads[1])
                self.tag_left = True
            if name == 'top':
                self.roads.append(roads[2])
                self.tag_top = True
            if name == 'bottom':
                self.roads.append(roads[3])
                self.tag_bottom = True

    def simulate_time_passage(self):
        # 判断是否过了1秒，如果是，增加一分钟
        start_time = self.start_time.second
        end_time = datetime.now().second
        hours = abs(int(end_time - start_time))
        self.current_time += timedelta(minutes=hours * 0.1)
        print(self.current_time)

    def get_state(self):
        agent_position = (self.agent.x, self.agent.y)  # 代理机器人的位置
        target_positions = self.target_position  # 目标位置

        state = {
            'agent_position': agent_position,
            'target_positions': target_positions
        }

        return state

    def binary_forward(self):
        move_x_distance = 1  # 默认移动距离
        move_y_distance = 1  # 默认移动距离
        distance_x_to_target = abs(self.agent.x - self.target_position[0])
        distance_y_to_target = abs(self.agent.y - self.target_position[1])

        if distance_x_to_target > self.width / 2:
            move_x_distance = int(distance_x_to_target / 8)  # 如果距离大于20，则调整移动距离
            move_x_distance = 10
        elif self.width / 2 > distance_x_to_target > self.width / 4:
            move_x_distance = int(distance_x_to_target / 2)
            move_x_distance = 10
        elif self.width / 4 > distance_x_to_target > self.width / 8:
            move_x_distance = int(distance_x_to_target / 2)
        elif self.width / 8 > distance_x_to_target > self.width / 16:
            move_x_distance = int(distance_x_to_target / 2)

        if distance_y_to_target > self.height / 2:
            move_y_distance = int(distance_y_to_target / 2)
            move_y_distance = 10
        elif self.height / 2 > distance_y_to_target > self.height / 4:
            move_y_distance = int(distance_y_to_target / 2)
            move_y_distance = 10
        elif self.height / 4 > distance_y_to_target > self.height / 8:
            move_y_distance = int(distance_y_to_target / 2)
        elif self.height / 8 > distance_y_to_target > self.height / 16:
            move_y_distance = int(distance_y_to_target / 2)

        return move_x_distance, move_y_distance

    def has_cache_item(self):
        if len(self.cache_items) > 0:
            print("len : " + str(len(self.cache_items)))

            # 使用原始列表的副本进行迭代
            for item in self.cache_items.copy():
                # 在原始列表上进行删除操作
                self.cache_items.remove(item)
                self.check_item(item.item_id, item.x, item.y, item.length, item.width, item.start_time,
                                item.processing_time, item.exit_time)
        else:
            print("列表为空。")

    def record_step(self, action, reward, done):
        step_info = {
            'action': action,
            'agent_position': (self.agent.x, self.agent.y),
            'target_position': self.target_position,
            'total_reward': self.total_reward,
            'elapsed_time': self.total_step_time,
            'conflict_count': self.conflict_count,
        }
        self.step_records.append(step_info)
        if done:
            self.save_records_to_csv()
            self.total_step_time = 0
            self.total_reward = 0
            self.conflict_count = 0

    def agent_move(self, action, move_x_distance, move_y_distance):
        # 执行动作并更新环境状态
        if action == 0:  # 代理机器人向上移动
            (self.agent.x, self.agent.y) = (self.agent.x, max(0, self.agent.y - move_y_distance))

            # time.sleep(0.001 * move_y_distance)  # 模拟移动的时间
        elif action == 1:  # 代理机器人向下移动
            (self.agent.x, self.agent.y) = (self.agent.x, min(self.height, self.agent.y + move_y_distance))
            if self.agent.y + self.agent.length > self.height:
                (self.agent.x, self.agent.y) = (self.agent.x, self.height - self.agent.length)
            # time.sleep(0.001 * move_y_distance)
        elif action == 2:  # 代理机器人向左移动
            (self.agent.x, self.agent.y) = (max(0, self.agent.x - move_x_distance), self.agent.y)

            # time.sleep(0.001 * move_x_distance)
        elif action == 3:  # 代理机器人向右移动
            (self.agent.x, self.agent.y) = (min(self.width, self.agent.x + move_x_distance), self.agent.y)

            # time.sleep(0.001 * move_x_distance)
        else:
            print("Invalid action!")
            reward = -100

    def get_earliest_item(self):
        if len(self.items) > 0 and self.target_position == (0, 0):
            # 获取最早出场时间的物品
            sorted_items = sorted(list(self.items.values()), key=lambda x: datetime.strptime(x.exit_time, "%Y/%m/%d"))
            min_date = sorted_items[0].exit_time
            filtered_items = list(filter(lambda x: x.exit_time == min_date, sorted_items))

            sorted_items = sorted(filtered_items, key=lambda x: x.time_remain)

            min_time_remain = sorted_items[0].time_remain
            sorted_items_remain = list(filter(lambda x: x.time_remain == min_time_remain, sorted_items))

            print("===================================================================================")
            print("sorted_items: ")
            for item in sorted_items_remain:
                print(item.item_id, item.exit_time)
            earliest_item = min(sorted_items_remain, key=lambda x: x.x)

            print("earliest_item: ", earliest_item.item_id, earliest_item.exit_time)
            print("current_time: ", self.current_time)
            if datetime.strptime(earliest_item.exit_time, "%Y/%m/%d") <= self.current_time:
                # 设置抽取的物品
                # self.item_random = earliest_item
                # self.item = self.item_random
                self.item = earliest_item
                # 获取目标位置
                self.task_positions.append((self.item.x, self.item.y))
                self.target_position = self.task_positions.pop(-1)
            else:
                pass

    def set_current_time(self, new_time):
        self.current_time = new_time

    def add_interfering_item(self):
        if self.arrive_interfering_position():
            item = self.interfering_items.pop(-1)
            print("干扰物品的是：", item.item_id)
            print("干扰物品的位置：", item.x, item.y)
            print("机器人到达要添加物品的位置：", item.x, item.y)
            self.check_item(item.item_id, 1, item.y, item.length, item.width, item.start_time,
                            item.processing_time,
                            item.exit_time)
            self.item = self.getInitItem()
            tmp_agent_x = self.agent.x
            tmp_agent_y = self.agent.y
            self.agent = self.item
            self.agent.x = tmp_agent_x
            self.agent.y = tmp_agent_y

    def arrive_interfering_position(self):
        if len(self.interfering_items) != 0 and self.agent_has_item is False and (self.agent.x, self.agent.y) == \
                (self.interfering_items[-1].x, self.interfering_items[-1].y):
            return True
        return False

    def conflict_resolve(self, reward):
        # 在代理机器人移动过程中检测冲突
        print("代理机器人的状态： ", self.agent.item_id, self.agent.x, self.agent.y)
        print("是否携带物品：", self.agent_has_item)
        if self.agent_has_item is True:
            for other_item in list(self.items.values()):
                if other_item.item_id.strip('agent_') != self.agent.item_id.strip(
                        'agent_') and self.check_collision(self.agent, other_item):
                    # 处理冲突
                    print(
                        f"冲突发生：代理机器人携带的物品与其他物品冲突  " + other_item.item_id.strip(
                            'agent_') + "     " +
                        self.agent.item_id.strip('agent_'))
                    # 随机选择一种处理方式
                    tag_conflict = self.conflict_count
                    random_action = choice(
                        [self.handle_conflict_1, self.handle_conflict_2, self.handle_conflict_3])

                    # 执行随机选择的处理方式
                    random_action(other_item)

                    # self.handle_conflict_2(other_item)
                    reward -= 3000 * (self.conflict_count - tag_conflict)

        return reward

    def is_one(self):
        try:
            if self.arrive_interfering_position():
                self.add_interfering_item()
                self.agent_has_item = False
                self.item = self.getInitItem()
                tmp_agent_x = self.agent.x
                tmp_agent_y = self.agent.y

                self.agent = self.item
                self.agent.x = tmp_agent_x
                self.agent.y = tmp_agent_y

                if not self.task_positions and not self.interfering_items:
                    self.task_positions.append((0, 0))
                    self.target_position = self.task_positions.pop(-1)
                else:
                    self.target_position = self.task_positions.pop(-1)
        except Exception as e:
            print(f"An error occurred in is_one: {e}")
            print("任务列表:", self.task_positions)
            print("self.agent；", self.agent.item_id, self.agent.x, self.agent.y)
            print("self.item: ", self.item.item_id, self.item.x, self.item.y)
            self.task_positions = []  # 清空任务列表

    def is_two(self):
        try:
            # 拿到的物品是否为空，如果为空，代表需要从场地捡一个物品携带
            if (self.agent_has_item == False) and (self.arrive_interfering_position() == False) and \
                    self.target_position != (0, 0) and (self.agent.x, self.agent.y) != (0, 0):
                item = self.items.get((self.target_position[0], self.target_position[1]))
                if item is None:
                    # 处理 item 为 None 的情况
                    print("------------------------item 为空--------------------")
                    if len(self.task_positions) == 0 and len(self.interfering_items) == 0:
                        self.target_position = (0, 0)
                    else:
                        self.target_position = self.task_positions.pop(-1)
                else:
                    self.item = item
                    self.agent = self.item
                    if self.remove_item(item):
                        self.agent_has_item = True
                    # this is need to code
                    self.choose_road(self.agent.x, item.y)
                    # self.task_positions.append((self.width, item.y))
                    self.target_position = self.task_positions.pop(-1)
        except Exception as e:
            print(f"An error occurred in is_two: {e}")
            print("任务列表:", self.task_positions)
            print("self.agent；", self.agent.item_id, self.agent.x, self.agent.y)
            print("self.item: ", self.item.item_id, self.item.x, self.item.y)
            print("item: ", item)

    def step(self, action):
        print("---------------------------------------------------------------")
        print("现在干涉物品的长度：  " + str(len(self.interfering_items)))
        # print("现在的物品有:")
        # for k, v in self.items.items():
        #     print(k, v.item_id)
        # for interfering in self.interfering_items:
        #     print("干涉物品是：  " + str(interfering.item_id) + "  " + str(interfering.x) + "  " + str(interfering.y))
        if self.agent_has_item:
            print("Agent has item!       " + str(self.agent.item_id))
        # 记录每一步的时间
        done = False
        step_time = datetime.now()
        #  检测是否有缓存的物品需要加入
        self.has_cache_item()
        # 快速移动
        move_x_distance, move_y_distance = self.binary_forward()
        # 奖励初始化
        reward = 0

        # -------------------分割线----------------------
        self.fix()

        if self.target_position == (0, 0):
            if len(self.items) == 0 and len(self.cache_items) == 0:
                if len(self.task_positions) == 0 and len(self.interfering_items) == 0:
                    done = True
                    reward = 10000
                    new_state = self.get_state()
                    self.total_step_time = 0
                    self.total_step_time = round(self.total_step_time, 5)
                    # 记录每一步的信息
                    self.record_step(action, reward, done)
                    return new_state, reward, done, {}
                done = False
                self.target_position = self.task_positions.pop(0)

            self.get_earliest_item()

        reward = self.conflict_resolve(reward)
        # 执行动作并更新环境状态
        self.agent_move(action, move_x_distance, move_y_distance)

        # -------------------分割线----------------------

        # 计算奖励
        if self.target_position != (0, 0) and self.target_position[0] < self.width:
            x_distance_to_target = abs(self.agent.x - self.target_position[0])
            y_distance_to_target = abs(self.agent.y - self.target_position[1])
            reward += 300.0 - x_distance_to_target - y_distance_to_target  # 根据距离计算奖励

        if self.agent.x > self.target_position[0]:
            reward -= 100
        if self.agent.y > self.target_position[1]:
            reward -= 100

        # -------------------分割线----------------------
        if (self.agent.x, self.agent.y) == self.target_position:
            if self.target_position[0] < self.width or self.target_position[1] < self.height:
                self.is_one()
                self.is_two()

            reward += 500  # 到达目标位置的奖励
            if self.agent.x >= self.width or self.agent.x <= 0 \
                    or self.agent.y <= 0 or self.agent.get_rectangle()[3] >= self.height:
                # 记录每一步的信息
                self.out_list.append(self.item)
                self.out_listed()
                self.item = self.getInitItem()
                tmp_agent_x = self.agent.x
                tmp_agent_y = self.agent.y

                self.agent = self.item
                self.agent.x = tmp_agent_x
                self.agent.y = tmp_agent_y

                self.agent_has_item = False
                reward += 800  # 成功搬运物品的奖励
                if len(self.interfering_items) == 0 and len(self.task_positions) == 0:
                    self.task_positions.append((0, 0))
                elif len(self.task_positions) != 0:
                    self.target_position = self.task_positions.pop(-1)
                else:
                    print("task_positions is null")

            if len(self.agent.item_id) == 11:
                self.agent.item_id = 'agent'
            if len(self.agent.item_id) <= 10:
                self.agent.item_id = 'agent_' + str(self.item.item_id)
            self.agent.color = 'red'

        else:
            done = False

        # reward = self.conflict_resolve(reward, action)
        # # 执行动作并更新环境状态
        # self.agent_move(action, move_x_distance, move_y_distance)

        if len(self.task_positions) > 0:
            print("任务位置的长度是：", len(self.task_positions))
            print("当前任务位置是：", self.target_position[0], self.target_position[1])
            print("下一步任务位置是：", self.task_positions[-1][0], self.task_positions[-1][1])

        if len(self.task_positions) > 0 and self.agent.x == self.target_position[0] \
                and self.agent.y == self.target_position[1]:
            self.target_position = self.task_positions.pop(-1)

        # -------------------分割线----------------------

        self.simulate_time_passage()
        # 更新状态
        new_state = self.get_state()
        self.clean_on_road()

        self.render()  # 更新环境
        # 更新环境
        if self.target_position == (0, 0) and len(self.task_positions) == 0 \
                and len(self.interfering_items) == 0:
            reward = 0
            print("---------------------------任务完成-------------------------")
            done = True
            new_state = self.get_state()
            reward += 1000
            # self.target_position = (0, 0)
            self.total_step_time = 0
            self.total_step_time = round(self.total_step_time, 5)
            self.record_step(action, reward, done)
            if not list(self.items.values()):
                pass
            else:
                earliest_item = min(list(self.items.values()), key=lambda x: datetime.strptime(x.exit_time, "%Y/%m/%d"))
                # # 获取最早出场时间的物品
                # sorted_items = sorted(list(self.items.values()), key=lambda x: x.exit_time)
                # sorted_items = sorted(sorted_items, key=lambda x: x.time_remain)
                # earliest_item = min(sorted_items, key=lambda x: x.x)

                # if datetime.strptime(earliest_item.exit_time, "%Y/%m/%d") == \
                #         datetime.strptime(str(datetime.strptime(self.current_time, "%Y/%m/%d")), "%Y/%m/%d"):
                #     pass
                # else:
                self.set_current_time(datetime.strptime(earliest_item.exit_time, "%Y/%m/%d"))
            return new_state, reward, done, {}  # 代理机器人到达目标位置，任务完成

        self.total_reward += reward
        self.total_step_time += (datetime.now() - step_time).total_seconds()
        self.total_step_time = round(self.total_step_time, 5)
        self.record_step(action, reward, done)
        return new_state, reward, done, {}

    def fix(self):
        if self.target_position == (0, 0) and len(self.agent.item_id) == 10 and self.agent_has_item is False:
            item = self.items.get((self.target_position[0], self.target_position[1]))
            self.item = item
            tmp_agent_x = self.agent.x
            tmp_agent_y = self.agent.y
            self.agent = self.item
            self.agent.x = tmp_agent_x
            self.agent.y = tmp_agent_y
            self.agent_has_item = True
            self.remove_item(item)
            self.choose_road(self.width, self.agent.y)
            self.target_position = self.task_positions.pop(-1)

    def out_listed(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        base_filename = f"{current_date}_out_list_records.csv"

        path = base_filename

        # Check if the file with today's date already exists

        with open(path, mode='w', newline='') as file:
            fieldnames = ['item_id', 'start_time', 'exit_time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for item in self.out_list:
                writer.writerow({
                    'item_id': item.item_id,
                    'start_time': item.start_time,
                    'exit_time': item.exit_time
                })

    def save_records_to_csv(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        base_filename = f"{current_date}_simulation_records.csv"

        path = base_filename

        # Check if the file with today's date already exists

        with open(path, mode='w', newline='') as file:
            fieldnames = ['action', 'agent_position', 'target_position', 'total_reward', 'elapsed_time',
                          'conflict_count']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for record in self.step_records:
                writer.writerow(record)

    def handle_conflict_1(self, interfering_item):
        """
        处理冲突的方式1：重新放置干涉方块
        """
        self.conflict_count += 2
        self.exchange_agent_item(interfering_item)

        print("冲突解决1： 现在的agent携带的物品是  " + self.agent.item_id.strip('agent_'))
        print("冲突解决1： 现在的Item携带的物品是  " + self.item.item_id.strip('agent_'))
        print("任务位置有： ")
        print(self.task_positions)
        # this is need to code
        self.choose_road(self.agent.x, self.agent.y)
        # self.task_positions.append((self.width, interfering_item.y))
        self.target_position = self.task_positions.pop(-1)

    def handle_conflict_2(self, interfering_item):
        """
        处理冲突的方式2：移动至相邻的上下行
        """
        self.conflict_count += 1
        self.agent.color = 'red'
        print("冲突解决2： 现在的agent携带的物品是  " + self.agent.item_id.strip('agent_'))
        print("冲突解决2： 现在的Item携带的物品是  " + self.item.item_id.strip('agent_'))

        target_row = self.get_target_row(interfering_item)

        if target_row is not None:
            # 移动干涉方块至相邻的上下行中
            self.move_to_target_row(interfering_item, target_row)
            # this is need to code
            self.choose_road(self.item.x, self.item.y)
            # self.task_positions.append((self.width, self.item.y))
            self.target_position = self.task_positions.pop(-1)
        else:
            self.handle_conflict_1(interfering_item)
        # 待目标方块搬出后，不将这些干涉方块放回原所在行

    def handle_conflict_3(self, interfering_item):
        """
        处理冲突的方式3：直接从邻行搬出
        """
        self.items.update({(self.agent.x, self.agent.y): self.agent})

        self.agent.color = 'red'
        print("冲突解决3： 现在的agent携带的物品是  " + self.agent.item_id.strip('agent_'))
        print("冲突解决3： 现在的Item携带的物品是  " + self.item.item_id.strip('agent_'))

        target_row = self.get_target_row(interfering_item)

        if target_row is not None:
            # 移动干涉方块至相邻的上下行中
            self.move_to_target_row(self.item, target_row)
            # 获取最后一个键（key）
            last_key = list(self.items.keys())[-1]

            # 获取最后一个物品
            last_item = self.items[last_key]
            self.item = last_item
            self.items.pop(last_key)
            self.agent = self.item
            # this is need to code
            self.choose_road(self.item.x, self.item.y)
            # self.task_positions.append((self.width, self.item.y))
            self.target_position = self.task_positions.pop(-1)
        else:
            self.handle_conflict_1(interfering_item)

    def move_to_target_row(self, item, target_row):
        self.remove_item(item)
        self.agent = self.item
        if self.agent.length - target_row < 0:
            self.check_item(item.item_id, 1, item.y - target_row, item.length, item.width, item.start_time,
                            item.processing_time, item.exit_time, item.time_remain)
        else:
            self.check_item(item.item_id, 1, item.y + target_row, item.length, item.width, item.start_time,
                            item.processing_time, item.exit_time, item.time_remain)

    def exchange_agent_item(self, interfering_item):
        """
        交换agent和interfering_item
        """
        self.task_positions.append((interfering_item.x, interfering_item.y))
        self.task_positions.append((self.agent.x, self.agent.y))
        self.agent.item_id = self.agent.item_id.strip('agent_')
        self.items.update({(self.agent.x, self.agent.y): self.agent})
        if len(self.interfering_items) == 0:
            tmp_interfering_item = copy.copy(interfering_item)
            self.interfering_items.append(tmp_interfering_item)
        else:
            for item in self.interfering_items:
                if interfering_item.item_id != item.item_id:
                    tmp_interfering_item = copy.copy(interfering_item)
                    self.interfering_items.append(tmp_interfering_item)
                    break
        self.remove_item(interfering_item)
        print("干扰物品位置： " + str(interfering_item.x), str(interfering_item.y))

        self.item = interfering_item
        self.agent = self.item
        self.agent_has_item = True

    def get_target_row(self, current_item):
        """
        获取目标方块上下行中的一个可用行

        Parameters:
            current_row (int): 当前行号

        Returns:
            target_row (int): 目标方块上下行中的一个可用行，如果没有可用行则返回 None
            :param current_item:
        """
        length = current_item.length

        # 检查移动到上面的行是否会发生冲突
        if current_item.length in self.segment_heights:
            length = current_item.length
            index = self.segment_heights.index(length)
        elif (current_item.length + 1) in self.segment_heights:
            length = current_item.length + 1
            index = self.segment_heights.index(length)
        elif (current_item.length + 2) in self.segment_heights:
            length = current_item.length + 2
            index = self.segment_heights.index(length)
        elif (current_item.length + 3) in self.segment_heights:
            length = current_item.length + 3
            index = self.segment_heights.index(length)
        elif (current_item.length + 4) in self.segment_heights:
            length = current_item.length + 4
            index = self.segment_heights.index(length)
        elif (current_item.length + 5) in self.segment_heights:
            length = current_item.length + 5
            index = self.segment_heights.index(length)
        else:
            length = 20
            index = self.segment_heights.index(length)
        upper_row = length
        print("upper_row: " + str(upper_row))
        if not self.is_conflict_with_target(upper_row,
                                            current_item, 0) and upper_row >= length and upper_row - length < 2 and \
                upper_row != 20:
            return upper_row

        # 检查移动到下面的行是否会发生冲突
        lower_row = self.segment_heights[index + 1]
        print("lower_row: " + str(lower_row))
        if not self.is_conflict_with_target(lower_row,
                                            current_item, 1) and lower_row >= length and lower_row - length < 2 and \
                lower_row != 8:
            return lower_row

        # 如果上面和下面的行都会发生冲突，则返回 None 表示没有可用行
        return None

    def is_conflict_with_target(self, target_row, current_item, action):
        """
        检查移动到目标行是否会发生冲突

        Parameters:
            target_row (int): 目标行号

        Returns:
            is_conflict (bool): 是否与目标发生冲突，如果发生冲突则为 True，否则为 False
            :param current_item:
            :param target_row:
            :param item:
        """
        # 在这里添加检查冲突的逻辑，例如检查与其他物体的碰撞等
        item = Item(current_item.item_id, current_item.x, current_item.y + target_row, current_item.length,
                    current_item.width,
                    current_item.start_time, current_item.processing_time, current_item.exit_time,
                    current_item.time_remain, current_item.color)

        is_conflict = self.check_collision(current_item, item)

        return is_conflict

    def reset(self):
        # Reset the environment to its initial state
        state = self.initial_state  # Reset the state to the initial state
        self.agent = Item('agent', 0, 0, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')

        self.target_position = (0, 0)  # 目标位置
        self.agent_has_item = False
        self.total_reward = 0
        self.total_step_time = 0
        self.item = Item('tmp', self.agent.x, self.agent.y, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')
        self.task_positions = []
        return state

    """
    添加物品到 环境当中
    """

    def add_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time, time_remain):
        if len(self.items) >= self.number:
            # 如果空地上的物品数量已经达到限制，你可以采取适当的操作，例如引发异常
            raise ValueError("空地上的物品数量已经达到限制")
        item_color = self.colors[len(self.items) % len(self.colors)]

        item = Item(item_id, x, y, length, width, start_time, processing_time, exit_time, time_remain, item_color)
        # print("add 方法中添加物品：  id ： " + item.item_id + "   " + item.start_time.__str__())
        return item

    # 在 remove_item 方法中添加如下行
    def remove_item(self, item):
        """
        从场地中移除物品
        """
        if item is None:
            return False
        position = (item.x, item.y)
        del self.items[position]
        return True

    def binary_search_insert(self, lst, value, type=0):
        # 二分查找插入元素到有序列表中
        left, right = 0, len(lst) - 1
        if right == -1:
            lst.insert(left, value)
            return
        while left <= right:
            mid = (left + right) // 2
            if lst[mid] == value:
                lst.insert(mid, value)
            elif lst[mid] < value:
                left = mid + 1
            else:
                right = mid - 1
        lst.insert(right, value)

    def divide_seg(self, height):
        if len(self.segment_heights) > 0 and len(self.y_positions) > 0:
            if (self.segment_heights[-1] + self.y_positions[-1]) >= self.height and self.seg_high_length <= height \
                    and self.seg_mid_length <= height and self.seg_low_length <= height:
                return

        for i in self.segment_heights:
            if i == height:
                return
        # 判断属于的段并进行二分查找插入
        if self.seg_mid <= height < self.seg_high and self.seg_high_length >= height:
            self.binary_search_insert(self.segment_high, height, 1)
            self.seg_high_length -= height
        elif self.seg_low < height < self.seg_mid and self.seg_mid_length >= height:
            self.binary_search_insert(self.segment_mid, height, 2)
            self.seg_mid_length -= height
        elif 0 < height <= self.seg_low and self.seg_low_length >= height:
            self.binary_search_insert(self.segment_low, height, 3)
            self.seg_low_length -= height

        self.segment_heights = []
        self.segment_high.sort(reverse=True)
        self.segment_heights.extend(self.segment_high)
        self.segment_mid.sort(reverse=True)
        self.segment_heights.extend(self.segment_mid)
        self.segment_low.sort(reverse=True)
        self.segment_heights.extend(self.segment_low)

    def initialize_segment(self, seg_high_length=74, seg_mid_length=58, seg_low_length=68, seg_high=20,
                           seg_mid=16, seg_low=12):
        self.seg_high = seg_high  #
        self.seg_mid = seg_mid
        self.seg_low = seg_low
        self.seg_high_length = seg_high_length - 20
        self.seg_mid_length = seg_mid_length
        self.seg_low_length = seg_low_length

    def check_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time, time_remain):
        """
        检查到达的物品对象。
        如果物品的到达时间早于当前时间，则将物品添加到环境中。
        如果物品的到达时间晚于当前时间，则将物品添加到缓存列表中。等到时间到达时再将物品添加到环境中。

        参数：
        - item: 到达的物品对象

        返回：
        - 无返回值
        """
        item = self.add_item(item_id, x, y, length, width, start_time, processing_time, exit_time, time_remain)
        # print(item_id + "  " + item.start_time.__str__())
        self.divide_seg(item.length)
        # 创建一个列表来存储y轴刻度的位置
        if self.tag_top:
            current_y = 20
        else:
            current_y = 1
        self.y_positions = []
        if self.tag_bottom and self.tag_top:
            self.height = 173
        #

        for i in range(len(self.segment_heights)):
            if len(self.segment_mid) > 0 and self.segment_heights[i] == self.segment_mid[0]:
                current_y = 92
            elif len(self.segment_low) > 0 and self.segment_heights[i] == self.segment_low[0]:
                current_y = 150
            self.y_positions.append(current_y)
            current_y += self.segment_heights[i]

        if self.current_time >= datetime.strptime(item.start_time, "%Y/%m/%d"):
            # print("添加物品时当前时间 :" + self.current_time.__str__())
            # print("进入物品的开始时间和 id:" + item.item_id + "   " + item.start_time.__str__())
            # 检查相同 y 坐标的物品
            if item.length in self.segment_heights:
                index = self.segment_heights.index(item.length)
            elif (item.length + 1) in self.segment_heights:
                index = self.segment_heights.index(item.length + 1)
            elif (item.length + 2) in self.segment_heights:
                index = self.segment_heights.index(item.length + 2)
            elif (item.length + 3) in self.segment_heights:
                index = self.segment_heights.index(item.length + 3)
            elif (item.length + 4) in self.segment_heights:
                index = self.segment_heights.index(item.length + 4)
            else:
                index = 0
            item.y = self.y_positions[index]
            # print("item.id:      " + item.item_id + "             " + str(item.y))
            # print("item.length:      " + str(self.segment_heights[index]) + "             " + str(item.length))
            com_y_items = self.filter_item_by_y(item.y)
            com_y_items.sort(key=lambda x: x.x, reverse=True)
            if com_y_items:
                # 如果存在相同 y 坐标的物品，则设置添加物品的 x 为前面物品的 x + width
                last_item = com_y_items[0]
                item.x = last_item.x + last_item.width + 1
            else:
                # 如果不存在相同 y 坐标的物品，则设置添加物品的 x
                item.x = item.x
            self.items[(item.x, item.y)] = item
        else:
            if len(self.cache_items) == 0:
                self.cache_items.append(item)
            else:
                if item not in self.cache_items:
                    self.cache_items.append(item)

    def filter_item_by_y(self, y):
        """
        过滤相同物品的 y 坐标。

        :return: items列表
        """
        items_com = []
        for (k, v) in self.items.items():
            if k[1] == y or k[1] + 1 == y or k[1] - 1 == y:
                items_com.append(v)
        return items_com

    def check_collision(self, item1, item2):
        epsilon = 0.1  # 误差
        """
        检查两个矩形是否相交。

        参数：
        - rectangle1: 第一个矩形的坐标 (left, top, right, bottom)
        - rectangle2: 第二个矩形的坐标 (left, top, right, bottom)

        返回：
        - 如果矩形相交，则返回 True，否则返回 False
        """
        rectangle1 = item1.get_rectangle()
        rectangle2 = item2.get_rectangle()
        # if not (rectangle1[2] < rectangle2[0] or  # 左
        #         rectangle1[0] > rectangle2[2] or  # 右
        #         rectangle1[3] < rectangle2[1] or  # 上
        #         rectangle1[1] > rectangle2[3]):
        #     return True
        # return False
        return (
                rectangle1[0] < rectangle2[2] and
                rectangle1[2] > rectangle2[0] and
                rectangle1[1] < rectangle2[3] and
                rectangle1[3] > rectangle2[1]
        )

    def clean_on_road(self):
        """
        清除道路上的物品
        :return: item_id
        """
        for (k, v) in self.items.items():
            if k[0] >= self.width or k[0] < 0 or v.get_rectangle()[3] >= self.height or k[1] < 0:
                item_id = v.item_id
                self.remove_item(v)
                return item_id

    def x_and_y(self):
        left_N = 0
        right_N = 0
        top_N = 0
        bottom_N = 0

        for i in range(len(self.roads)):
            if 'left' == self.roads[i]['name']:
                left_N = i
            if 'right' == self.roads[i]['name']:
                right_N = i
            if 'top' == self.roads[i]['name']:
                top_N = i
            if 'bottom' == self.roads[i]['name']:
                bottom_N = i
        # 创建 x 轴刻度标签
        if self.tag_left:
            x_ticks = [-self.roads[left_N]['width'], 0, self.width]
        else:
            x_ticks = [0, self.width]

        # 绘制刻度标签
        plt.xticks(x_ticks, fontsize=8)
        self.change_bias()
        # if left_road 存在 加 bias
        if self.tag_left:
            self.change_bias()
            plt.xlim(-self.bias, self.width + self.bias)
        if self.tag_right:
            plt.xlim(0, self.width + 20)
        if self.tag_left and self.tag_right:
            plt.xlim(-20, self.width + 20)
        else:
            plt.xlim(0, self.width)
        # plt.ylim(-20, self.height)
        # if 存在 上下道路 加 bias
        if self.tag_top and self.tag_bottom:
            self.change_bias()
            plt.imshow(np.ones((self.height, self.width)), cmap='binary', interpolation='none',
                       origin='upper')
        else:
            plt.imshow(np.ones((self.height, self.width)), cmap='binary', interpolation='none',
                       origin='upper')
        if self.tag_top:
            self.change_bias()
            plt.imshow(np.ones((self.height + self.bias, self.width)), cmap='binary', interpolation='none',
                       origin='upper')
        if self.tag_bottom:
            self.change_bias()
            plt.imshow(np.ones((self.height + self.bias, self.width)), cmap='binary', interpolation='none',
                       origin='upper')
        # 创建一个列表来存储y轴刻度的位置
        # if self.tag_top:
        #     current_y = 20
        # else:
        #     current_y = 0
        # y_positions = []
        # if self.tag_bottom and self.tag_top:
        #     self.height = 173
        #
        # for i in range(len(self.segment_heights)):
        #     y_positions.append(current_y)
        #     current_y += self.segment_heights[i]

        plt.yticks(self.y_positions, self.segment_heights, fontsize=8)

        return right_N, left_N, top_N, bottom_N

    def render(self):
        plt.figure(figsize=(5, 5))
        list_num = self.x_and_y()

        # Draw agent
        agent_x = self.agent.x
        agent_y = self.agent.y
        agent_width = self.agent.width
        agent_length = self.agent.length
        agent_color = self.agent.color

        agent_rect = plt.Rectangle((agent_x, agent_y), agent_width, agent_length, color=agent_color, alpha=0.5)
        plt.gca().add_patch(agent_rect)

        # 添加文本 "box1" 到方块内部
        text_x = agent_x + agent_width / 2
        text_y = agent_y + agent_length / 2
        plt.text(text_x, text_y, self.agent.item_id, ha='center', va='center', fontsize=6, color='black')

        for (x, y), item in self.items.items():
            rect = plt.Rectangle((x, y), item.width, item.length, color=item.color, alpha=0.5)
            plt.gca().add_patch(rect)
            # 添加文本 "box1" 到方块内部
            text_x = x + item.width / 2
            text_y = y + item.length / 2
            plt.text(text_x, text_y, item.item_id, ha='center', va='center', fontsize=6, color='black')

        self.draw_road(list_num[0], list_num[1], list_num[2], list_num[3])

        plt.show()

    def draw_road(self, right_N, left_N, top_N, bottom_N):
        if self.tag_right:
            # 右
            road_width_right = self.roads[right_N]['width']
            road_color_right = self.roads[right_N]['color']
            road_rect_right = plt.Rectangle((self.width, 0), road_width_right, self.height,
                                            color=road_color_right,
                                            alpha=1)
            # 绘制道路
            plt.gca().add_patch(road_rect_right)
        if self.tag_left:
            # 左
            road_width_left = self.roads[left_N]['width']
            road_color_left = self.roads[left_N]['color']
            road_rect_left = plt.Rectangle((- road_width_left, 0), road_width_left, self.height,
                                           color=road_color_left,
                                           alpha=1)
            plt.gca().add_patch(road_rect_left)

        if self.tag_top:
            # 上
            road_height_top = self.roads[top_N]['height']
            road_color_top = self.roads[top_N]['color']

            road_rect_top = plt.Rectangle((0, 0), self.width, road_height_top, color=road_color_top,
                                          alpha=1)

            plt.gca().add_patch(road_rect_top)

        if self.tag_bottom:
            # 下
            road_height_bottom = self.roads[bottom_N]['height']
            road_color_bottom = self.roads[bottom_N]['color']
            road_rect_bottom = plt.Rectangle((-20, self.height), self.width + 40,
                                             road_height_bottom,
                                             color=road_color_bottom,
                                             alpha=1)
            plt.gca().add_patch(road_rect_bottom)

    def getInitItem(self):
        if self.tag_top:
            init_item = Item('agent', 0, 20, 5, 5, '2017/9/1', 0, '2017/9/1', 0, 'black')
        else:
            init_item = Item('agent', 0, 0, 5, 5, '2017/9/1', 0, '2017/9/1', 0, 'black')
        return init_item


def main():
    # 创建环境实例
    env = WarehouseEnvironment(width=75, height=153, number=50)
    env.setRoads('top', 'bottom')

    # 示例用法：添加物品并显示环境
    env.check_item('B001', 0, 114, 11, 8, '2017/9/1', 13, '2017/9/22')
    env.render()
    env.check_item('B003', 8, 114, 11, 8, '2017/9/2', 13, '2017/9/22')
    env.check_item('B007', 19, 114, 11, 8, '2017/9/2', 13, '2017/9/29')
    env.check_item('B009', 40, 114, 11, 8, '2017/9/2', 13, '2017/9/27')
    env.check_item('B0013', 60, 114, 11, 8, '2017/9/2', 13, '2017/9/21')
    env.render()


if __name__ == "__main__":
    main()
