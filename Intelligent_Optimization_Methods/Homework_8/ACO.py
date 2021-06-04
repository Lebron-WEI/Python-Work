import numpy as np
import math
import random
import matplotlib.pyplot as plt

class ACO:#蚁群算法

    def __init__(self, alpha, beta, Q, m, rho, maxstep):
        self.alpha = alpha
        self.beta = beta
        self.m = m #蚂蚁数量
        self.Q = Q
        self.rho = rho #信息素挥发因子
        self.maxstep = maxstep

        self.v = 269 #背包重量上限
        self.index = np.array([0,1,2,3,4,5,6,7,8,9], dtype = np.int) #物品编号
        self.w = np.array([95,4,60,32,23,72,80,62,65,46], dtype = np.float) #物品重量
        self.p = np.array([55,10,47,5,4,50,8,61,85,87], dtype = np.float) #物品价值
        self.n = len(self.w)

        self.table = np.zeros((self.m, self.n), dtype = np.int)
        self.route_best = np.zeros((self.maxstep, self.n))
        self.value_best = np.zeros((self.maxstep, 1))
        self.tau = np.ones((self.n, self.n), dtype = np.float)

    def eta(self): #计算eta函数，采用价值/重量
        ans = self.p/self.w
        return ans

    def run(self):
        eta = self.eta()
        step  = 0
        while step < self.maxstep:
            start = np.zeros((self.m, 1), dtype = np.int)
            for i in range(self.m): #为m个蚂蚁选择起始拿的物品的编号
                start[i][0] = random.randint(0, self.n - 1)
            self.table[:,0] = start[:,0] #将start的值赋给table的第0列

            for i in range(self.m):
                for j in range(1,self.n):
                    if self.table[i][j-1] == 999: #如果上一次没拿物品，那么这次也不会拿物品
                        self.table[i][j] = 999
                        continue

                    has_visited_index = self.table[i, 0:j] #已经拿过的物品的编号
                    allow_index = np.delete(self.index, has_visited_index) #允许选取的物品的编号

                    temp = 0
                    for k in range(len(has_visited_index)):
                        temp = temp + self.w[has_visited_index[k]]

                    too_heavy_index=np.array([], dtype=np.int)
                    for l in range(len(allow_index)):
                        if temp + self.w[allow_index[l]] > self.v: #若物品l超重，则从允许列表中删去
                            too_heavy_index = np.append(too_heavy_index, [l])

                    allow_index = np.delete(allow_index, too_heavy_index)


                    if len(allow_index) == 0: #如果没有物品可拿，就不执行下面的语句了，进入下一个循环
                        self.table[i][j] = 999
                        continue


                    P = np.zeros(len(allow_index)) #有物品可拿时，计算允许列表中拿的下一个物品是allow_index[t]的概率
                    for t in range(len(allow_index)):
                        P[t] = ((self.tau[self.table[i][j-1]][allow_index[t]])**self.alpha)*((eta[allow_index[t]])**self.beta)

                    P = P/sum(P)
                    r = random.random()
                    s = 0
                    temp1 = 0
                    for z in range(len(P)):
                        s += P[z]
                        if s > r:
                            temp1 = z
                            break
                    target_index = allow_index[temp1]
                    self.table[i][j] = target_index

            value = np.zeros((self.m,1))
            for i in range(self.m):
                route = self.table[i,:]
                for j in range(self.n):
                    if route[j] != 999:
                        value[i][0] = value[i][0] + self.p[route[j]]

            temp_value = 0
            temp_index = 0
            for i in range(self.m):
                if value[i][0] > temp_value:
                    temp_value = value[i][0]
                    temp_index = i
            max_value = temp_value
            max_index = temp_index

            if step == 0:
                self.value_best[step][0] = max_value
            else:
                if max_value < self.value_best[step-1][0]:
                    self.value_best[step,0] = self.value_best[step-1,0]
                    self.route_best[step,:] = self.route_best[step-1,:]
                else:
                    self.value_best[step,0] = max_value
                    self.route_best[step,:] = self.table[max_index,:]

            step = step + 1

            d_tau = np.zeros((self.n, self.n))
            for i in range(self.m):
                for j in range(self.n):
                    if self.table[i][j] != 999 and self.table[i][j+1] != 999:
                        d_tau[self.table[i][j]][self.table[i][j+1]] = d_tau[self.table[i][j]][self.table[i][j+1]] + self.Q*value[i]
            self.tau = (1-self.rho)*self.tau + d_tau
            self.table = np.zeros((self.m, self.n), dtype=np.int)

        print("The worst result:", np.min(self.value_best))
        print("The best result:", np.max(self.value_best))
        print("Mean result:", np.mean(self.value_best))
        result_x = np.arange(0, self.maxstep, 1)
        result_y = self.value_best
        plt.xlim(0, 150)  # 限定横轴的范围
        plt.ylim(0, 320)  # 限定纵轴的范围
        plt.plot(result_x, result_y, 'r-', label="price")
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"step") #X轴标签
        plt.ylabel(u"price") #Y轴标签

        plt.show()
        plt.close(0)

if __name__ == "__main__":
    alpha = 0.5
    beta = 0.5
    Q = 1
    m = 3
    rho = 0.1
    maxstep = 150
    aco = ACO(alpha, beta, Q, m, rho, maxstep)
    aco.run()
