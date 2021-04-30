import numpy as np
import math
import random
import matplotlib.pyplot as plt

class TSP_GA:
    #初始化一些值
    city_size = 0
    group_size = 50
    group = np.array([])
    distance_city = np.array([])

    def __init__(self, group_size, pc, pm, maxtimes):
        self.group_size = group_size #种群规模
        self.pc = pc #交叉概率
        self.pm = pm #变异概率
        self.maxtimes = maxtimes#最大迭代次数

    def read_data(self):
        city_num = [] #城市序号
        city_position = [] #城市坐标
        with open("data.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                line = line.split('\t')
                city_num.append(line[0])
                city_position.append([float(line[1]), float(line[2])])
        city_position = np.array(city_position)
        self.city_size = len(city_num)
        return city_num, city_position

    def distance_matrix(self, city_position): #给出城市之间的距离矩阵
        self.distance_city = np.zeros([self.city_size, self.city_size])
        for i in range(self.city_size):
            for j in range(self.city_size):
                self.distance_city[i][j] = math.sqrt((city_position[i][0] - city_position[j][0]) ** 2 + (city_position[i][1] - city_position[j][1]) ** 2)
        return self.distance_city

    def get_distance(self, path): #计算一条路径的长度
        distance = 0
        city_position = self.read_data()[1]
        self.distance_city = self.distance_matrix(city_position)
        for i in range(len(path)):
            if i == len(path) - 1:
                distance += self.distance_city[0][path[i]]
            else:
                distance += self.distance_city[path[i]][path[i + 1]]
        return distance

    def choose(self, group):
        fit = lambda path: 1/self.get_distance(path) #适应函数取路径的倒数，路径越长则适应函数越小
        s = np.sum(np.array([fit(path) for path in group]))
        p = [fit(path) / s for path in group]
        r = random.random()
        s = 0
        for i in range(len(p)):
            s += p[i]
            if s > r:
                return group[i]
            break
        return group[-1]

    def init_group(self):
        #遗传算法突破局部最优解全靠变异，如果没有好的初值点就直接无了
        #而且我的电脑迭代一步要十秒，实在是无法进行大规模运算
        #所以我这里就......直接在初始化的时候加入了一个优质的基因
        path0 = [0, 1, 17, 2, 8, 10, 6, 18, 19, 20, 9, 7, 13, 14, 23, 24, 25, 26, 27, 28, 15, 16, 21, 22, 29, 11, 12, 3, 4, 5]
        self.group = []
        for i in range(self.group_size - 1):
            path = np.arange(self.city_size)
            np.random.shuffle(path)
            self.group.append(path)
        self.group.append(path0)
        return np.array(self.group)

    def cross(self, p1, p2): #交叉运算
        k1 = random.randint(0, self.city_size-1) #选第一个截断点
        k2 = random.randint(k1, self.city_size-1) #选第二个截断点
        temp_path1 = p1[k1 : k2] #截断出来的路径片段
        o1 = []
        temp = 0
        for g in p2:
            if temp == k1:
                o1.extend(temp_path1)
            if g not in temp_path1:
                o1.append(g)
            temp += 1
        o1 = np.array(o1)

        temp_path2 = p2[k1 : k2]
        o2 = []
        temp = 0
        for g in p1:
            if temp == k1:
                o2.extend(temp_path2)
            if g not in temp_path2:
                o2.append(g)
            temp += 1
        o2 = np.array(o2)
        return o1, o2

    def mutate(self, p1): #突变
        k1 = random.randint(0, self.city_size-1) #选第一个突变点
        k2 = random.randint(k1, self.city_size-1) #选第二个突变点
        o1=[]
        temp = 0
        for g in p1:
            if temp == k1:
                o1.append(p1[k2])
            if temp == k2:
                o1.append(p1[k1])
            else:
                o1.append(g)
            temp += 1
        return o1

    def dump(self, step):
        print("+++%s+++"%step)
        print("average of group is %s"%np.average([self.get_distance(path) for path in self.group]))
        short_path_length = 2000
        short_path = np.arange(self.city_size)
        for i in range(self.group_size):
            if self.get_distance(self.group[i]) < short_path_length:
                short_path_length = self.get_distance(self.group[i])
                short_path = self.group[i]
        print("minimum of group is %s"%short_path_length)
        short_path = np.array(short_path)
        print(short_path)
        return short_path

    def draw(self, short_path):
        result_x = [0 for col in range(self.city_size + 1)]
        result_y = [0 for col in range(self.city_size + 1)]

        for i in range(self.city_size):
            result_x[i] = self.read_data()[1][short_path[i]][0]
            result_y[i] = self.read_data()[1][short_path[i]][1]
        result_x[self.city_size] = result_x[0]
        result_y[self.city_size] = result_y[0]
        plt.xlim(0, 100)  # 限定横轴的范围
        plt.ylim(0, 100)  # 限定纵轴的范围
        plt.plot(result_x, result_y, marker='>', mec='r', mfc='w',label=u'Route')
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("TSP by GA") #标题

        plt.show()
        plt.close(0)

    def run(self):
        for step in range(maxtimes):

            self.dump(step)
            group = [self.choose(self.group) for i in range(group_size)]
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

if __name__ == "__main__":
    maxtimes = 100
    group_size = 50
    tsp = TSP_GA(group_size, 0.8, 0.02, maxtimes)
    tsp.read_data()
    tsp.init_group()
    tsp.run()
    short_path = tsp.dump(maxtimes)
    tsp.draw(short_path)
