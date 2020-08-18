# -*- coding:utf-8 -*-
# @Time : 2020/8/17 10:37
# @Author: PCF
# @File : DQN_agent.py
import numpy as np
from tensorflow import keras
from gym import Env
from collections import deque
from agent.Agent import Agent
import random


class DQNAgent(Agent):
    """
    使用深度Q网络进行更新的智能体
    """

    def __init__(self, env: Env, mem_size=2000):
        """
        构造函数

        :param env: 环境
        :param mem_size: 智能体记忆大小
        """
        super().__init__(env)

        self.action_dim = env.action_space.n  # 行为空间的维数为行为动作的种类数
        # 获取环境状态空间的维度，分为一维离散的Discrete和连续的Box两种类型
        self.state_dim = 1 if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]

        self.model = self._create_model()  # 近似Q函数的网络模型
        self.target_model = self._create_model()  # 目标Q网络加速收敛
        self.train_times = 0  # 模型训练的次数
        self.update_freq = 200  # 更新目标网络的频率，每训练多少次更新一次

        self.mem_size = mem_size  # 记忆容量
        self.mem_queue = deque(maxlen=self.mem_size)  # 使用队列对智能体的经验进行记忆

    def _create_model(self):
        """
        创建深度Q网络模型
        :return: 网络模型
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(100, input_dim=self.state_dim, activation='relu'))
        model.add(keras.layers.Dense(self.action_dim, activation='relu'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))
        return model

    def policy(self, state, use_epsilon, episode):
        """
        根据贪心策略在行为空间中选取动作 TODO 支持连续动作空间

        :param state: 当前的状态
        :param use_epsilon: true:采用epsilon贪心策略 false:采用贪心策略
        :param episode: 当前的episode数
        :return: 选择的动作
        """
        epsilon = 1.0 / (episode + 1)
        if not use_epsilon or random.random() > epsilon:
            return np.argmax(self.model.predict(np.array([state]))[0])
        else:
            return self.action_space.sample()

    def save_model(self, path='models/DQN.h5'):
        """
        存储网络模型
        :param path: 存储路径
        """
        self.model.save(path)

    def load_model(self, path='models/DQN.h5'):
        """
        加载网络模型
        :param path: 模型文件路径
        """
        self.model = keras.models.load_model(path)

    def _remember(self, state, action, next_state, reward):
        """
        记录经历 TODO 为接近目标的经历提供额外奖励，加速收敛
        :param state: 状态
        :param action: 在state下执行的行为
        :param next_state: 在state下执行action后到达的状态
        :param reward: 在state下执行action所获得的奖励
        :return:
        """

        self.mem_queue.append((state, action, next_state, reward))

    def _model_train(self, batch_size=64, alpha=0.99, gamma=0.95):
        """
        训练网络模型

        :param batch_size: 训练的batch大小
        :param alpha: 学习率
        :param gamma: 折扣
        """
        if len(self.mem_queue) < self.mem_size:
            return

        self.train_times += 1
        # 更新目标网络参数
        if self.train_times % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        # 在记忆中随机采样经历来训练网络
        mem_batch = random.sample(self.mem_queue, batch_size)
        state_batch = np.array([train[0] for train in mem_batch])
        next_state_batch = np.array([train[2] for train in mem_batch])

        # 利用已有的模型估计当前状态和下一状态的状态动作价值函数值
        q_state = self.model.predict(state_batch)
        q_next_state = self.target_model.predict(next_state_batch)

        # 利用下一状态的估计Q值更新当前状态的Q值
        for i, mem in enumerate(mem_batch):
            _, action, _, reward = mem
            q_state[i][action] = q_state[i][action] + alpha * (
                    reward + gamma * np.argmax(q_next_state[i]) - q_state[i][action])

        self.model.fit(state_batch, q_state, verbose=0)  # 利用更新后的Q值来对模型进行训练

    def learning(self, gamma=0.95, alpha=0.99, max_episode=500, render_episode=-1):
        """
        深度Q网络学习算法
        :param gamma: 折扣
        :param alpha: 学习率
        :param max_episode: 最大训练episode数
        :param render_episode: 开始渲染环境所需要的episode数，渲染会严重降低训练速度
        """
        for cur_episode in range(max_episode):
            state = self.obs = self.env.reset()
            is_done = False
            rewards_pre_episode = 0
            while not is_done:
                if cur_episode > render_episode:
                    self.env.render()
                action = self.policy(state, True, cur_episode)
                next_state, reward, is_done, info = self.action(action)
                rewards_pre_episode += reward

                self._remember(state, action, next_state, reward)  # 记住经历
                self._model_train(alpha=alpha, gamma=gamma)  # 训练模型

                state = next_state

            print("episode {0} is {1:.1f} rewards.".format(cur_episode, rewards_pre_episode))
            self.rewards.append(rewards_pre_episode)
