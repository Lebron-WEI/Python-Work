import numpy as np
import math
import random
import matplotlib.pyplot as plt

class QP_GA: #二次规划的GA算法
    def __init__(self, group_size, pc, pm, maxtimes):
        self.group = []
        self.group_size = group_size #种群规模
        self.pc = pc #交叉概率
        self.pm = pm #变异概率
        self.maxtimes = maxtimes#最大迭代次数
        self.max_tar_function = np.zeros(self.maxtimes, dtype = np.float)
        self.n = 250
        self.table = np.zeros((self.n, self.n), dtype = np.float)

    def read_data(self): #读取表格数据
        with open("7-1.txt", 'r') as f:
            lines = f.readlines()
            for line in lines[1:len(lines)]:
                line = line.split('\n')[0]
                line = line.split(' ')
                self.table[int(line[1])-1][int(line[2])-1] = line[3]
                self.table[int(line[2])-1][int(line[1])-1] = line[3]
        return self.table

    def tar_function(self, t): #目标函数
        ans1 = np.dot(t, self.table)
        ans = np.dot(ans1, t.T)
        return ans

    def fit_function(self, x_): #适应函数越大越好
        ans = np.exp(self.tar_function(x_)/500)
        return ans

    def choose(self, group): #选择函数，轮盘赌
        s = np.sum(np.array([self.fit_function(x_) for x_ in group]))
        p = [self.fit_function(x_) / s for x_ in group]
        r = random.random()
        temp = 0
        num = 0
        for i in range(len(p)):
            temp += p[i]
            if temp > r:
                num = i
                break
        return group[num]

    def init_group(self): #随机初始化种群
        for g in range(self.group_size):
            x = np.zeros(self.n, dtype = int)
            for i in range(self.n):
                x[i] = random.randint(0, 1)
            self.group.append(x[:])
        return self.group

    def cross(self, p1, p2): #交叉运算
        k1 = random.randint(0, self.n-1) #选第一个截断点
        k2 = random.randint(k1, self.n-1) #选第二个截断点

        temp_path1 = p1[k1 : k2].copy() #截断出来的路径片段
        temp_path2 = p2[k1 : k2].copy()

        o1 = p1.copy()
        o2 = p2.copy()
        o2[k1 : k2] = temp_path1
        o1[k1 : k2] = temp_path2
        return o1, o2

    def mutate(self, p1): #突变
        k1 = random.randint(0, self.n - 1) #选一个突变点
        p1[k1] = 1 - p1[k1]
        return p1

    def dump(self, step): #输出函数
        print("+++%s+++"%step)
        print("average of group is %s"%np.average([self.tar_function(x_) for x_ in self.group]))
        self.max_tar_function[step] = self.tar_function(self.group[0])
        #max_x = self.group[0][:]
        for i in range(self.group_size):
            if self.tar_function(self.group[i]) > self.max_tar_function[step]:
                self.max_tar_function[step] = self.tar_function(self.group[i])
                #max_x = self.group[i][:]
        print("maximum of group is %s"%self.max_tar_function[step])
        #print(max_x)
        return self.max_tar_function

    def draw(self): #画图函数
        x0 = [i for i in range(self.maxtimes)]
        #plt.xlim(0, 10000)  # 限定横轴的范围
        plt.ylim(0, 50000)  # 限定纵轴的范围
        plt.plot(x0, self.max_tar_function, 'r-', label='Tar_Fun')
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("tar_fun") #标题
        plt.show()

    def run(self):
        for step in range(self.maxtimes):

            self.dump(step)
            group = []
            for i in range(self.group_size):
                group.append(self.choose(self.group))
            random.shuffle(group)
            self.group = []

            kc = math.floor(len(group)*self.pc // 2)
            km = math.floor(len(group)*self.pm)

            for i in range(kc):
                new = self.cross(group[i*2], group[i*2+1])
                self.group += new

            for i in range(km):
                new = (self.mutate(group[kc*2+i]))
                self.group.append(new)

            for i in range(len(group)-kc*2-km):
                self.group.append(group[kc*2+km+i])

        print("in all, the maximum is %s"%np.max(self.max_tar_function))

if __name__ == "__main__":
    maxtimes = 5000
    group_size = 100
    qp = QP_GA(group_size, 0.8, 0.02, maxtimes)
    qp.read_data()
    qp.init_group()
    qp.run()
    qp.draw()