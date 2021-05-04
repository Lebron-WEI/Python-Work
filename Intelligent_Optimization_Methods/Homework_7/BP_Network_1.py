import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Function_Approximation: #函数逼近
    def __init__(self, hidden_layer_size, lr):

        self.hidden_layer_size = hidden_layer_size #隐藏层的神经元个数
        self.lr = lr #学习率

        self.W1 = np.random.random((100, hidden_layer_size)) #输入层与隐藏层之间的权重矩阵
        self.B1 = np.random.random((hidden_layer_size, 1)) #隐藏层神经元的阈值
        self.W2 = np.random.random((hidden_layer_size, 100)) #隐藏层与输出层之间的权重矩阵
        self.B2 = np.random.random((100, 1)) #输出层神经元的阈值

    def tar_fun(self, x_): #目标函数f(x)=sin(pi*x)
        ans = math.sin(math.pi * x_)
        return ans

    def sigmoid(self, x_): #激活函数选择sigmoid
        return 1 / (1 + np.exp(-x_))

    def sigmoid_grad(self, x_): #激活函数导数
        return (1.0 - self.sigmoid(x_)) * self.sigmoid(x_)

    def generate_data(self, fun, x_start, x_stop, x_num): #生成数据
        x = np.linspace(x_start, x_stop, x_num)[:, np.newaxis]
        y = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            y[i] = fun(x[i])
        return x, y

    def run_network(self, x_):
        hidden = self.sigmoid(np.dot(self.W1.T, x_) + self.B1)
        out = self.sigmoid(np.dot(self.W2.T, hidden) + self.B2)
        return out

    def loss(self, x_, t): #选择均方误差作为损失函数
        y_ = self.run_network(x_)
        return y_, np.mean((t - y_) ** 2)

    def train(self, x, target):
        hidden = self.sigmoid(np.dot(self.W1.T, x) + self.B1)
        out = self.sigmoid(np.dot(self.W2.T, hidden) + self.B2)

        erro = out * (1 - out) * (target - out)
        errhidden = hidden * (1 - hidden) * np.matmul(self.W1.T, erro)
        self.W1 += self.lr * np.matmul(x, errhidden.T)
        self.W2 += self.lr * np.matmul(hidden, erro.T)
        self.B1 += self.lr * errhidden
        self.B2 += self.lr * erro

if __name__ == "__main__":
    hidden_layer_size = 3
    lr = 1
    max_steps = 100
    network = Function_Approximation(hidden_layer_size, lr)

    ans = int(input("please input number: ('0' means untrained output, '1' means trained output)"))
    if ans == 0:
        x0, y0 = network.generate_data(network.tar_fun, 0, 1, 100)
        y1, loss = network.loss(x0, y0)
        plt.plot(x0, y0, 'r-', label='Tar_Fun')
        plt.plot(x0, y1, 'b-,', label='My_Fun')
        plt.text(0.5, 0, 'Loss=%.4f' % abs(loss), fontdict={'size': 10, 'color': 'red'})
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("untrained output") #标题
        plt.show()

    elif ans == 1:
        x0, y0 = network.generate_data(network.tar_fun, 0, 1, 100)
        for k in range(max_steps):
            network.train(x0,y0)
        y2, loss = network.loss(x0, y0)
        plt.plot(x0, y0, 'r-', label='Tar_Fun')
        plt.plot(x0, y2, 'b-,', label='My_Fun')
        plt.text(0.5, 0, 'Loss=%.4f' % abs(loss), fontdict={'size': 10, 'color': 'red'})
        plt.legend()  # 让图例生效
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"x") #X轴标签
        plt.ylabel(u"y") #Y轴标签
        plt.title("trained output") #标题
        plt.show()
    else:
        print("wrong number")