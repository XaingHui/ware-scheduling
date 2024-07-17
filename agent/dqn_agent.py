import csv

import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random

from env.envv import WarehouseEnvironment


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()
        tensorflow.keras.utils.disable_interactive_logging()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # def choose_action(self, state, agent_position, target_position, count):
    #     if np.random.rand() <= self.epsilon and count > 0:
    #         return np.random.choice(self.action_size)
    #
    #     # 计算机器人当前位置和目标位置之间的水平和垂直距离
    #     distance_x = target_position[0] - agent_position[0]
    #     distance_y = target_position[1] - agent_position[1]
    #
    #     # 获取当前Q值
    #     q_values = self.model.predict(state)[0]
    #     print("Q值：", q_values)
    #
    #     # 根据距离选择行动
    #     if abs(distance_x) > abs(distance_y):
    #         # 在水平方向上的距离较大，选择左移或右移
    #         if distance_x > 0:
    #             return 3 if q_values[3] > q_values[2] else 2  # 右移或左移
    #         else:
    #             return 2 if q_values[2] > q_values[3] else 3  # 左移或右移
    #     else:
    #         # 在垂直方向上的距离较大，选择上移或下移
    #         if distance_y > 0:
    #             return 1 if q_values[1] > q_values[0] else 0  # 下移或上移
    #         else:
    #             return 0 if q_values[0] > q_values[1] else 1  # 上移或下移

    def choose_action(self, state, agent_position, target_position, count):
        if np.random.rand() <= self.epsilon and count > 0:
            return np.random.choice(self.action_size)
        # 计算机器人当前位置和目标位置之间的水平和垂直距离
        distance_x = target_position[0] - agent_position[0]
        distance_y = target_position[1] - agent_position[1]
        # 获取当前Q值
        # 根据距离选择行动
        if abs(distance_x) > abs(distance_y):
            # 在水平方向上的距离较大，选择左移或右移
            if distance_x > 0:

                return 3  # 右移
            else:
                return 2  # 左移
        else:
            # 在垂直方向上的距离较大，选择上移或下移
            if distance_y > 0:
                return 1  # 下移
            else:
                return 0  # 上移

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        import tensorflow as tf
        # 在适当的位置调用TensorFlow的清理函数
        tf.keras.backend.clear_session()

        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# def add_items_from_csv(env, csv_file):
#     with open(csv_file, 'r') as file:
#         csv_reader = csv.reader(file)
#         next(csv_reader)  # 跳过 CSV 文件的标题行
#
#         for row in csv_reader:
#             item_id = row[0]
#             x = int(row[1])
#             y = int(row[2])
#             length = int(row[3])
#             width = int(row[4])
#             start_time = str(row[5])
#             processing_time = int(row[6])
#             exit_time = str(row[7])
#
#             # 添加物品到环境
#             env.check_item(item_id, x, y, length, width, start_time, processing_time, exit_time)

def add_items_from_csv(env, csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过 CSV 文件的标题行

        for row in csv_reader:
            item_id = row[0]
            x = int(row[1])
            y = int(row[2])
            length = int(row[3])
            width = int(row[4])
            start_time = str(row[6])
            exit_time = str(row[7])
            processing_time = 5
            if row[12] == '':
                time_remian = 0
            else:
                time_remian = int(row[12])

            # 添加物品到环境
            env.check_item(item_id, x, y, length, width, start_time, processing_time, exit_time, time_remian)


def main():
    env = WarehouseEnvironment(width=250, height=200, number=90, time='2024/4/25')
    env.setRoads('right')
    env.initialize_segment(92, 58, 70, 20, 16, 12)

    add_items_from_csv(env, '../data_change/segment01.csv')
    env.render()
    state_size = len(env.get_state())
    action_size = 4  # 代表上移、下移和不执行动作
    agent = DQNAgent(state_size, action_size)

    episodes = 100

    for episode in range(episodes):
        state = env.get_state()  # Get the initial state
        agent_position = np.array(list(state['agent_position']))
        target_position = np.array(list(state['target_positions']))
        # item_positions = np.array(list(state['item_positions']))

        # 将这些位置信息合并成一个数组
        state_array = np.concatenate([agent_position, target_position])
        #  print(state_array)
        state = np.reshape(state_array, [-1, state_size])
        # print(state)
        total_reward = 0
        done = False
        count = 5
        while not done:
            action = agent.choose_action(state, agent_position, target_position, count)
            next_state, reward, done, _ = env.step(action)
            print(next_state)
            agent_position = np.array(list(next_state['agent_position']))
            target_position = np.array(list(next_state['target_positions']))
            # item_positions = np.array(list(state['item_positions']))

            # 将这些位置信息合并成一个数组
            next_state_array = np.concatenate([agent_position, target_position])
            next_state = np.reshape(next_state_array, [-1, state_size])

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            count -= 1
            # if len(env.items) == 0:
            #     total_reward = 10000
            #     break
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
