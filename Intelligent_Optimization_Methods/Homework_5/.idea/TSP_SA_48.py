import numpy as np
import math
import random
import matplotlib.pyplot as plt

class TSP_SA:#TSP问题，模拟退火算法
    #初始化一些值
    city_size = 0
    distance_city = np.array([])

    def __init__(self, T_final, alpha, inner_co):
        self.T_final = T_final #截止温度
        self.alpha = alpha #退温系数
        self.inner_co = inner_co #内循环系数 #其中：内循环次数=内循环系数*城市数目

    def read_data(self):
        city_num = [] #城市序号
        city_position = [] #城市坐标
        with open("att48.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                line = line.split('\t')
                city_num.append(int(line[0])-1)
                city_position.append([float(line[1]), float(line[2])])
        city_position = np.array(city_position, dtype = np.float32)
        self.city_size = len(city_num)
        self.city_num = city_num[:]
        self.city_position = city_position[:]
        self.inner_loop = self.inner_co*self.city_size
        return self.city_num, self.city_position

    def distance_matrix(self): #给出城市之间的距离矩阵
        self.distance_city = np.zeros([self.city_size, self.city_size])
        for i in range(self.city_size):
            for j in range(self.city_size):
                self.distance_city[i][j] = math.sqrt(((self.city_position[i][0] - self.city_position[j][0]) ** 2 + (self.city_position[i][1] - self.city_position[j][1]) ** 2))
        return self.distance_city

    def get_distance(self, path): #计算一条路径的长度
        distance = 0
        for i in range(len(path)):
            if i == len(path) - 1:
                distance += self.distance_city[0][path[i]]
            else:
                distance += self.distance_city[path[i]][path[i + 1]]
        return distance

    def nearest_city(self, current_city, cand_cities): #找离当前城市最近的城市
        temp_min = 200000
        next_city = None
        for i in range(len(cand_cities)):
            temp_dis = self.distance_city[current_city][cand_cities[i]]
            if temp_dis < temp_min:
                temp_min = temp_dis
                next_city = cand_cities[i]
        return next_city, temp_min

    def greedy_initial_path(self):
        current_city = 0
        cand_cities = self.city_num
        initial_path_length = 0
        initial_path = []
        initial_path.append(current_city)
        while len(cand_cities) > 1:
            cand_cities.remove(current_city)
            initial_path.append(self.nearest_city(current_city, cand_cities)[0]) #将下一个城市添加到路径列表中
            initial_path_length += self.nearest_city(current_city, cand_cities)[1] #将这段路径长度加入总路径长度中
            current_city = self.nearest_city(current_city, cand_cities)[0] #更新当前城市
        initial_path_length += self.distance_city[current_city][0] #回到起点
        return initial_path, initial_path_length

    def swap_2_cities(self, path):
        new_path = []

        k1 = random.randint(1, self.city_size-1) #选第一个交换点
        k2 = random.randint(k1, self.city_size-1) #选第二个交换点

        temp = 0
        for c in path:
            if temp == k1:
                new_path.append(path[k2])
            elif temp == k2:
                new_path.append(path[k1])
            else:
                new_path.append(c)
            temp += 1
        return new_path

    def get_T_start(self):
        fval0 = np.array([0 for i in range(100)])
        list0 = [i for i in range(self.city_size)]
        for i in range(100):
            random.shuffle(list0)
            fval0[i] = self.get_distance(list0)

        T_0 = -(np.max(fval0)-np.min(fval0))/np.log(0.9)
        return T_0

    def simulated_annealing(self, T_0):
        (path_now, length_now) = self.greedy_initial_path()#贪婪算法生成初始解
        self.best_so_far_path = path_now[:]
        self.best_so_far_length = length_now

        while(T_0 > self.T_final):

            print("Current temperature is %s"%T_0)

            for j in range(self.inner_loop):

                new_path = self.swap_2_cities(path_now)[:]
                temp_dis = self.get_distance(new_path)
                if temp_dis <= length_now:
                    path_now = new_path[:]
                    length_now = temp_dis
                if temp_dis > length_now:
                    delta_dis = temp_dis - length_now
                    if random.random() < np.exp(-delta_dis/T_0):
                        path_now = new_path[:]
                        length_now = temp_dis
                if temp_dis <= self.best_so_far_length:
                    self.best_so_far_path = new_path[:]
                    self.best_so_far_length = temp_dis

            print("Minimum path is %s"%self.best_so_far_path)
            print("Minimum length is %s"%self.best_so_far_length)
            T_0 = self.alpha * T_0#降温

        final_path = self.best_so_far_path
        final_length = self.best_so_far_length
        return final_path, final_length

    def draw(self, final_path):
        result_x = [0 for col in range(self.city_size + 1)]
        result_y = [0 for col in range(self.city_size + 1)]

        for i in range(self.city_size):
            result_x[i] = self.read_data()[1][final_path[i]][0]
            result_y[i] = self.read_data()[1][final_path[i]][1]
        result_x[self.city_size] = result_x[0]
        result_y[self.city_size] = result_y[0]
        plt.xlim(0, 8000)  # 限定横轴的范围
        plt.ylim(0, 8000)  # 限定纵轴的范围
        plt.plot(result_x, result_y, marker='>', mec='r', mfc='w',label=u'Route')
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("TSP by SA") #标题

        plt.show()
        plt.close(0)

if __name__ == "__main__":
    T_final = 0.01
    alpha = 0.9
    inner_co = 200
    tsp = TSP_SA(T_final, alpha, inner_co)
    tsp.read_data()
    tsp.distance_matrix()
    T_0 = tsp.get_T_start()
    final_path = tsp.simulated_annealing(T_0)[0]
    tsp.draw(final_path)
