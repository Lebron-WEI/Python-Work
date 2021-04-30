import numpy as np
import math
import random

class GA:
    def __init__(self, func, accfunc, enl, pc, pm, maxg):
        self.func = func
        self.accfunc = lambda x: (accfunc(x[0]), accfunc(x[1]))
        self.enl = enl
        self.pc = pc
        self.pm = pm
        self.maxg = maxg

    def choose(self, arr):
        fit = lambda x: math.e - self.func(x[0], x[1]) if self.func(x[0], x[1]) < math.e else 0
        s = np.sum(np.array([fit(self.accfunc(x)) for x in arr]))
        p = [fit(self.accfunc(x)) / s for x in arr]
        r = random.random()
        s = 0
        for i in range(len(p)):
            s += p[i]
            if s > r:
                return arr[i]
        return arr[-1]

    def init(self, num):
        self.group = [(random.randint(0, 2**self.enl-1), random.randint(0, 2**self.enl-1)) for i in range(num)]

    def cross(self, x, y):
        k = random.randint(0, self.enl-2)
        mask = 2**(self.enl-k-1)-1
        tmp = x & mask
        x &= ~mask
        x |= y & mask
        y &= ~mask
        y |= tmp
        return x, y

    def mutate(self, x):
        k = random.randint(0, self.enl-1)
        mask = 2**(self.enl-k-1)
        return x ^ mask

    def dump(self, step):
        print("+++%s+++"%step)
        print("average of group is %s"%np.average([self.func(self.accfunc(x)[0], self.accfunc(x)[1]) for x in self.group]))
        print("minimum of group is %s"%np.min([self.func(self.accfunc(x)[0], self.accfunc(x)[1]) for x in self.group]))

    def run(self):
        np = len(self.group)
        for step in range(maxg):

            self.dump(step)

            group = [self.choose(self.group) for i in range(np)]
            random.shuffle(group)
            self.group = []

            k1 = math.floor(len(group)*self.pc // 2)
            k2 = math.floor(len(group)*self.pm)

            for i in range(k1):
                newx_ = self.cross(group[i*2][0], group[i*2+1][0])
                newy_ = self.cross(group[i*2][1], group[i*2+1][1])
                new = [(newx_[0], newx_[1]), (newy_[0], newy_[1])]
                self.group += new

            for i in range(k2):
                new = (self.mutate(group[k1*2+i][0]), self.mutate(group[k1*2+i][1]))
                self.group.append(new)

            for i in range(len(group)-k1*2-k2):
                self.group.append(group[k1*2+k2+i])

if __name__ == "__main__":

    def F(x, y):
        return math.e-20*np.exp(-0.2*(x**2/2+y**2/2)**(0.5))-np.exp((math.cos(2*math.pi*x)+math.cos(2*math.pi*y))/2)

    maxg = 100
    ga = GA(F, lambda x: x / 63 * 10 - 5, 6, 0.8, 0.02, maxg)
    ga.init(30)
    ga.run()
