#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 10:53
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : PerQ_agent.py
# @Software: PyCharm
import logging

import gym
import torch
from tqdm import trange

from agent.DQN_agent import DQNAgent
from common.prioritized_memory import PrioritizedMemory


class PerQAgent(DQNAgent):
    """
    有限经验回放机制的DQN智能体
    """

    def __init__(self, rl_env: gym.Env, memory_size: int, batch_size: int,
                 update_pred: int, epsilon_decay: float, max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1, gamma: float = 0.99, path=None,
                 alpha: float = 0.2, beta: float = 0.6, prior_eps: float = 1e-6):
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
        :param alpha: 控制使用多少优先级
        :param beta: 控制使用多少重要性采样
        :param prior_eps: 控制生成多少采样
        """
        super(PerQAgent, self).__init__(rl_env, memory_size, batch_size, update_pred, epsilon_decay, max_epsilon,
                                        min_epsilon, gamma, path)

        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedMemory(self.state_shape, memory_size, batch_size, alpha)

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

        # 优先级采样需要beta来计算权重
        samples = self.memory.sample(beta=self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 平均之前进行重要性采样
        elementwise_loss = self.compute_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        # 反向传播，网络参数优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss

    def compute_loss(self, samples) -> torch.Tensor:
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
        loss = self.loss_func(q_value, target, reduction='none')
        return loss

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

                    # 增大参数beta
                    fraction = min(cur_episode / max_episode, 1.0)
                    self.beta = self.beta + fraction * (1.0 - self.beta)

                    loss = self.train_net()
                    if loss != -1:
                        loss = loss.data.item()
                        self.writer.add_scalar('loss', loss, global_step=self.step)  # 记录本次网络训练的损失值

                self.writer.add_scalar('score', score, global_step=cur_episode)  # 记录分数
                logging.debug('episode: {},score:{}'.format(cur_episode, score))
                t.set_postfix(loss=loss, score=score)  # 进度条相关信息

            if self.path:
                self.save_net(self.path)  # 存储参数
