import numpy as np
import math
import random
import matplotlib.pyplot as plt

X_1=np.array([[-1,1,-2,0]]).T
X_2=np.array([[-1,0,1.5,-0.5]]).T
X_3=np.array([[-1,-1,1,0.5]]).T

X=np.append(X_1,X_2,axis=1)
X=np.append(X,X_3,axis=1)#X是4*3的矩阵

Y=np.array([[-1,-1,1]])#Y是1*3的矩阵

W=(np.random.random([4,1])-0.5)*2#初始化权值，范围是[-1,1]
#W是4*1的矩阵

lr=0.11#设置学习率为0.11

output = np.array([[0,0,0]])#神经网络的输出(output)初始化为0

def iteration():#迭代函数
	global X,Y,W,lr
	output = np.sign(np.dot(W.T, X))
	delta = lr*(np.dot(X, (Y-output).T))
	W = W + delta#对权值矩阵进行调整

for i in range(100):
    print("第 %s 次迭代"%i)
    print(W)
    iteration()
    output=np.sign(np.dot(W.T, X))#计算当前的输出
    if(output==Y).all():
        print("Finished")
        break