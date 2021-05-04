import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Function_Approximation: #函数逼近
    def __init__(self, hidden_layer_size, lr):

        self.hidden_layer_size = hidden_layer_size #隐藏层的神经元个数
        self.lr = lr #学习率

        self.W1 = np.array([[0.2]*hidden_layer_size]*2) #输入层与隐藏层之间的权重矩阵
        self.B1 = np.array([[-0.1]*1]*hidden_layer_size) #隐藏层神经元的阈值
        self.W2 = np.array([[-0.2]*2]*hidden_layer_size) #隐藏层与输出层之间的权重矩阵
        self.B2 = np.array([[0.3]*1]*2) #输出层神经元的阈值

    def tar_fun(self, x_): #目标函数f(x)=sin(pi*x)
        ans = math.sin(math.pi * x_)
        return ans

    def sigmoid(self, x_): #激活函数选择sigmoid
        return 1 / (1 + np.exp(-x_))

    def sigmoid_grad(self, x_): #激活函数导数
        return (1.0 - self.sigmoid(x_)) * self.sigmoid(x_)

    def generate_data(self): #生成数据
        x = np.array([[1],[0]])
        y = np.array([[0],[1]])
        return x, y

    def run_network(self, x_):
        hidden = self.sigmoid(np.dot(self.W1.T, x_) + self.B1)
        out = self.sigmoid(np.dot(self.W2.T, hidden) + self.B2)
        return out

    def loss(self, x_, t): #选择均方误差作为损失函数
        y_ = self.run_network(x_)
        return y_, np.mean((t - y_) ** 2)

    def train(self, x, target):
        print("输入是%s"%x)

        hidden_in = np.dot(self.W1.T, x) + self.B1
        print("隐含神经元的净输入是%s"%hidden_in)

        hidden = self.sigmoid(hidden_in)
        print("隐含神经元的净输出是%s"%hidden)

        out_in = np.dot(self.W2.T, hidden) + self.B2
        print("输出神经元的净输入是%s"%out_in)

        out = self.sigmoid(out_in)
        print("输出神经元的实际输出是%s"%out)

        print("期待输出是%s"%target)

        y_, loss = self.loss(x, target)
        print("误差是%s"%loss)

        erro = out * (1 - out) * (target - out)
        errhidden = hidden * (1 - hidden) * np.matmul(self.W1.T, erro)
        self.W1 += self.lr * np.matmul(x, errhidden.T)
        self.W2 += self.lr * np.matmul(hidden, erro.T)
        self.B1 += self.lr * errhidden
        self.B2 += self.lr * erro

if __name__ == "__main__":
    hidden_layer_size = 3
    lr = 1
    max_steps = 3
    network = Function_Approximation(hidden_layer_size, lr)

    x0, y0 = network.generate_data()
    for k in range(max_steps):
        print("+++step %s+++"%k)
        network.train(x0,y0)