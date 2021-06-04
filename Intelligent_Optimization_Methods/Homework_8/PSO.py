import numpy as np
import math
import random
import matplotlib.pyplot as plt


class PSO():
	# PSO参数设置
    def __init__(self, pN, dim, max_iter):

        self.x_city = np.array([18, 22, 58, 71, 83, 91, 24, 18], dtype=float)
        self.y_city = np.array([54, 60, 69, 71, 46, 38, 42, 40], dtype=float)
        self.g_city = np.array([0, 0.89, 0.14, 0.28, 0.33, 0.21, 0.41, 0.57], dtype=float)
        self.q_car = 1. #车的货运量
        self.num_car = 3 #车的数量

        self.city_size = len(self.x_city) #城市的数量，包括0号城市（车站）
        self.distance_city = np.zeros((self.city_size, self.city_size)) #初始化距离矩阵

        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN #粒子数量
        self.dim = dim #搜索维度
        self.max_iter = max_iter #迭代次数
        #初始化粒子的位置和速度
        self.Xv = np.zeros((self.pN, self.dim)) #完成该任务车辆的编号k
        self.Xr = np.zeros((self.pN, self.dim)) #该任务在k车行驶路径中的次序r
        self.Vv = np.zeros((self.pN, self.dim))
        self.Vr = np.zeros((self.pN, self.dim))
        self.Xv_pbest = np.zeros((self.pN, self.dim)) #个体经历的最佳位置
        self.Xr_pbest = np.zeros((self.pN, self.dim))
        self.Xv_gbest = np.zeros((1, self.dim)) #全局最佳位置
        self.Xr_gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN) #每个个体的历史最佳适应值
        self.fit = 1e10 #全局最佳适应值

    #目标函数Sphere函数
    def function(self, Xv, Xr):

    def distance_matrix(self): #给出城市之间的距离矩阵
        for i in range(self.city_size):
            for j in range(self.city_size):
                self.distance_city[i][j] = math.sqrt((self.x_city[i] - self.x_city[j]) ** 2 + (self.y_city[i] - self.y_city[j]) ** 2)
        print(self.distance_city)

    #初始化种群
    def init_Population(self):
        for i in range(self.pN):      #因为要随机生成pN个数据，所以需要循环pN次
            for j in range(self.dim):      #每一个维度都需要生成速度和位置，故循环dim次
                self.Xv[i][j] = random.randint(1, self.num_car)
                self.Xr[i][j] = random.uniform(1, self.city_size - 1)
                self.Vv[i][j] = random.randint(- self.num_car + 1, self.num_car - 1)
                self.Vr[i][j] = random.uniform(- self.city_size + 2, self.city_size - 2)
            self.Xv_pbest[i] = self.Xv[i]
            tmp = self.function(self.Xv[i], self.Xr[i])  #得到现在最优
            self.p_fit[i] = tmp    #这个个体历史最佳的位置
            if tmp < self.fit:   #得到现在最优和历史最优比较大小，如果现在最优大于历史最优，则更新历史最优
                self.fit = tmp
                self.Xv_gbest = self.Xv[i]
                self.Xr_gbest = self.Xr[i]

    # 更新粒子位置
    def iterator(self):

if __name__ == "__main__":
    pso = PSO(pN=40, dim=7, max_iter=200)