#乘同余法中的底数a
a=3
#乘同余法中的模数m=2^k
k=10
m=pow(2,k)
#负指数分布中的/lambda
t=2
#逆变法产生负指数分布的随机数函数
import math
def f_reverse(u):
    return -(1/t)*math.log(u)

x=1
print(f_reverse(x/m))
x=(a*x) %m
while x != 1:
    print(f_reverse(x/m))
    x=(a*x) %m