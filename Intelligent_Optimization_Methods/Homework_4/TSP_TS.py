import numpy as np
import math
import random
import matplotlib.pyplot as plt

class TSP_TS:
    #初始化一些值
    city_size = 30
    distance_city = np.array([])
    tabu_list = []
    best_so_far_length = 2000
    best_so_far_path = []

    def __init__(self, candiate_path_size, tabu_size, maxtimes):
        self.candiate_path_size = candiate_path_size #候选路径规模
        self.tabu_size = tabu_size #禁忌表长度
        self.maxtimes = maxtimes#最大迭代次数

    def read_data(self):
        city_num = [] #城市序号
        city_position = [] #城市坐标
        with open("data.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                line = line.split('\t')
                city_num.append(int(line[0]))
                city_position.append([float(line[1]), float(line[2])])
        city_position = np.array(city_position, dtype = np.float32)
        self.city_size = len(city_num)
        self.city_num = city_num[:]
        self.city_position = city_position[:]
        return self.city_num, self.city_position

    def distance_matrix(self): #给出城市之间的距离矩阵
        self.distance_city = np.zeros([self.city_size, self.city_size])
        for i in range(self.city_size):
            for j in range(self.city_size):
                self.distance_city[i][j] = math.sqrt((self.city_position[i][0] - self.city_position[j][0]) ** 2 + (self.city_position[i][1] - self.city_position[j][1]) ** 2)
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
        temp_min = 2000
        next_city = None
        for i in range(len(cand_cities)):
            temp_dis = self.distance_city[current_city][cand_cities[i]]
            if temp_dis < temp_min:
                temp_min = temp_dis
                next_city = cand_cities[i]
        return next_city, temp_min

    def greedy_initial_path(self):
        current_city = 0
        cand_cities = self.city_num[:]
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
        return new_path, sorted([path[k1], path[k2]])

    def generate_new_path(self, path):
        candidate_path = [] #候选路径的集合
        candidate_path_length = [] #候选路径对应的长度的集合
        candidate_swap = [] #交换方式的集合
        new_path=[]
        while len(candidate_path) < self.candiate_path_size:
            if self.swap_2_cities(path)[1] not in candidate_swap: #此次生成新路径的交换方式不曾出现在交换方式的集合中
                candidate_path.append(self.swap_2_cities(path)[0])
                candidate_swap.append(self.swap_2_cities(path)[1])
                candidate_path_length.append(self.get_distance(self.swap_2_cities(path)[0]))

        min_path_length = 2000
        num_of_min_path = 0
        for i in range(len(candidate_path)):
            #print(self.get_distance(candidate_path[i]))
            if self.get_distance(candidate_path[i]) < min_path_length:
                min_path_length = self.get_distance(candidate_path[i])
                num_of_min_path = i

        if min_path_length < self.best_so_far_length: #如果此次交换集的最优值比历史最优值更好，则更新历史最优值和最优路线
            self.best_so_far_length = min_path_length
            self.best_so_far_path = candidate_path[num_of_min_path][:]
            new_path = candidate_path[num_of_min_path][:]
            if candidate_swap[num_of_min_path] in self.tabu_list:
                self.tabu_list.remove(candidate_swap[num_of_min_path]) #破禁
            elif len(self.tabu_list) >= self.tabu_size:
                self.tabu_list.remove(self.tabu_list[0])
            self.tabu_list.append(candidate_swap[num_of_min_path])

        else: #此次交换集未找到更优路径，则选择交换方式未在禁忌表中的次优
            t = 0
            while t < len(self.tabu_list) + 2:

                temp_min_path_length = 2000
                temp_num_of_min_path = 0
                for i in range(len(candidate_path)):
                    if self.get_distance(candidate_path[i]) < temp_min_path_length:
                        temp_min_path_length = self.get_distance(candidate_path[i])
                        temp_num_of_min_path = i

                if candidate_swap[temp_num_of_min_path] not in self.tabu_list:
                        self.tabu_list.append(candidate_swap[temp_num_of_min_path])
                        new_path = candidate_path[temp_num_of_min_path][:]
                        if len(self.tabu_list) > self.tabu_size:
                            self.tabu_list.remove(self.tabu_list[0])
                        break
                else:
                    candidate_path_length.remove(candidate_path_length[temp_num_of_min_path])
                    candidate_swap.remove(candidate_swap[temp_num_of_min_path])
                    candidate_path.remove(candidate_path[temp_num_of_min_path])
                    t += 1
        return new_path

    def tabu_search(self):
        (self.best_so_far_path, self.best_so_far_length) = self.greedy_initial_path()#贪婪算法生成初始解
        new_path = self.best_so_far_path[:]
        print(new_path)

        for step in range(self.maxtimes):
            print("+++%s+++"%step)
            new_path = self.generate_new_path(new_path)[:]
            print("minimum path is %s"%self.best_so_far_path)
            print("minimum length is %s"%self.best_so_far_length)

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
        plt.xlim(0, 100)  # 限定横轴的范围
        plt.ylim(0, 100)  # 限定纵轴的范围
        plt.plot(result_x, result_y, marker='>', mec='r', mfc='w',label=u'Route')
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("TSP by TS") #标题

        plt.show()
        plt.close(0)

if __name__ == "__main__":
    candiate_path_size = 200
    tabu_size = 50
    maxtimes = 2000
    tsp = TSP_TS(candiate_path_size, tabu_size, maxtimes)
    tsp.read_data()
    tsp.distance_matrix()
    final_path = tsp.tabu_search()[0]
    tsp.draw(final_path)
