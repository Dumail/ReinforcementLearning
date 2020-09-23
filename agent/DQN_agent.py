#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 10:22
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : DQN_agent.py
# @Software: PyCharm
import logging
from typing import Dict

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from agent.Agent import Agent
from common.memory import Memory
from common.networks import NetThreeLayer


class DQNAgent(Agent):
    """
    使用Nature DQN算法进行学习的智能体
    """

    def __init__(self, rl_env: gym.Env, memory_size: int, batch_size: int,
                 update_pred: int, epsilon_decay: float, max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1, gamma: float = 0.99, path=None):
        """
        构造函数
        :param rl_env: 环境
        :param memory_size: 经验池容量大小
        :param batch_size: 每次采样大小
        :param update_pred: 目标网络更新频率
        :param epsilon_decay: epsilon衰减率
        :param max_epsilon: 最大epsilon
        :param min_epsilon: 最小epsilon
        :param gamma: 折扣衰减
        :param path: 网络存储路径 None则不存储
        """
        super(DQNAgent, self).__init__(rl_env)
        self.state_shape = rl_env.observation_space.shape or rl_env.observation_space.n
        self.action_shape = rl_env.action_space.shape or rl_env.action_space.n
        self.memory = Memory(np.prod(self.state_shape), memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.update_pred = update_pred
        self.gamma = gamma
        self.step = 0  # 网络训练步数，用于目标网络更新
        self.path = path

        # 自动选择gpu或cpu
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        logging.info('Use device is {}'.format(self.device))

        # 优化网络与目标网络
        self.net = self.create_net()
        self.target_net = self.create_net()
        self.target_net.load_state_dict(self.net.state_dict())  # 参数相同
        self.target_net.eval()  # 目标网络设为评估模式，不需要反向传播

        self.optimizer = torch.optim.Adam(self.net.parameters())  # 优化器
        self.loss_func = torch.nn.functional.mse_loss

        # tensorboard记录器
        self.writer = SummaryWriter(log_dir='../logs/{}/'.format(self.__class__.__name__))

    def create_net(self):
        """
        创建神经网络
        :return:网络模型
        """
        return NetThreeLayer(self.state_shape, self.action_shape, self.device).to(self.device)

    def add_graph(self):
        """
        记录图结构
        """
        # 可视化网络模型
        with self.writer:
            samples = self.memory.sample()
            states = torch.FloatTensor(samples["obs"]).to(self.device)
            # writer.add_graph(self.net, (torch.FloatTensor(self.env.reset()).to(self.device),))
            self.writer.add_graph(self.net, states)

    def policy(self, state: np.ndarray, is_test=False) -> np.ndarray:
        """
        智能体选择动作的策略，epsilon贪心
        :param state: 智能体状态
        :param is_test: 是否是测试模式，该模式使用纯贪心
        :return: 选择的动作
        """

        if not is_test and self.epsilon > np.random.random():
            action = self.env.action_space.sample()  # 随机采样
        else:
            action = self.net(torch.FloatTensor(state).to(self.device)).argmax()  # 通过网络选择值最大的行为
            action = action.detach().cpu().numpy()

        if not is_test:
            self.epsilon = max(self.min_epsilon,
                               self.epsilon - (self.max_epsilon - self.min_epsilon)
                               * self.epsilon_decay)
        return action

    def action(self, action: np.ndarray):
        return self.env.step(action)

    def compute_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        通过样本计算网络的损失值
        :param samples: 采样得到的样本
        :return: 损失值
        """
        # 获取从采样中得到的各类数据并装换为Tensor
        states = torch.FloatTensor(samples["obs"]).to(self.device)
        next_states = torch.FloatTensor(samples["next_obs"]).to(self.device)
        actions = torch.LongTensor(samples["actions"].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = torch.LongTensor(samples["done"].reshape(-1, 1)).to(self.device)

        q_value = self.net(states).gather(1, actions)  # 状态行为对的当前价值
        next_q_value = self.compute_next_q_value(next_states)
        target = (rewards + self.gamma * next_q_value * (1 - done)).to(self.device)  # 计算TD目标
        loss = self.loss_func(q_value, target)
        return loss

    def compute_next_q_value(self, next_states: torch.FloatTensor):
        """
        计算下一状态的最大q值表
        :param next_states:
        :return:
        """
        return self.target_net(next_states).max(dim=1, keepdim=True)[0].detach()  # 下一状态中最大的q值

    def train_net(self):
        """
        采样经验用于网络训练
        :return: 本次训练的损失值
        """
        if len(self.memory) < self.batch_size:
            return -1  # 经验不够，为满足采样要求
        self.step += 1
        if self.step % self.update_pred == 0:
            self.target_net.load_state_dict(self.net.state_dict())  # 更新目标网络参数

        samples = self.memory.sample()
        loss = self.compute_loss(samples)

        # 反向传播，网络参数优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_net(self, path):
        """
        存储网络参数
        :param path: 参数文件路径
        """
        torch.save(self.net.state_dict(), path)
        logging.debug('Saved net')

    def load_net(self, path):
        """
        加载网络参数
        :param path: 参数文件路径
        """
        self.net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        logging.debug('Loaded net')

    def learning(self, max_episode: int):
        """
        训练智能体
        :param max_episode: 最大episode数
        """
        score = 0
        loss = 0

        with trange(max_episode) as t:
            for cur_episode in t:
                t.set_description("%s train episode: %d" % (self.__class__.__name__, cur_episode))  # 设置进度条标题

                score = 0
                state = self.env.reset()  # 初始化每个episode的状态
                done = False
                while not done:
                    action = self.policy(state)  # 根据状态选择行为
                    next_state, reward, done, _ = self.action(action)  # 执行行为

                    self.memory.store(state, action, reward, next_state, done)  # 存入一次经验

                    state = next_state
                    score += reward

                    loss = self.train_net()
                    if loss != -1:
                        loss = loss.data.item()
                        self.writer.add_scalar('loss', loss, global_step=self.step)  # 记录本次网络训练的损失值

                self.writer.add_scalar('score', score, global_step=cur_episode)  # 记录分数
                logging.debug('episode: {},score:{}'.format(cur_episode, score))
                t.set_postfix(loss=loss, score=score)  # 进度条相关信息

            self.add_graph()
            if self.path:
                self.save_net(self.path)  # 存储参数

    def test(self, max_episode: int = 1, render=False):
        """
        测试智能体
        :param: test_num 测试次数
        :param: render 是否渲染环境
        :return: 平均分数
        """
        scores = np.zeros((max_episode, 1), dtype=float)  # 每次测试的分数数组
        with trange(max_episode) as t:
            for cur_episode in t:
                score = 0
                state = self.env.reset()
                done = False
                while not done:
                    if render:
                        self.env.render()  # 渲染环境
                    action = self.policy(state, is_test=True)
                    next_state, reward, done, _ = self.action(action)

                    state = next_state
                    score += reward

                scores[cur_episode] = score
        logging.debug('Test scores are: {}'.format(scores))
        return np.average(scores)
