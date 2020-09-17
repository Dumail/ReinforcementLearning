# -*- coding:utf-8 -*-
# @Time : 2020/8/1 16:56
# @Author: PCF
# @File : vis.py
from matplotlib import pyplot as plt


def show(agent, max_episode):
    """
    绘制TD算法智能体训练中，每个episode获得奖励的折线图
    :param agent: TD算法智能体
    :param max_episode: 训练最大episode数，对应智能体中应含相同个数的reward
    """
    episode = [i + 1 for i in range(max_episode)]
    plt.plot(episode, agent.rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
