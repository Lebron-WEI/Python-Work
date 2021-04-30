#乘同余法中的底数a
a=3
#乘同余法中的模数m=2^k
k=4
m=pow(2,k)

x=1
print(x/m)
x=(a*x) %m
while x != 1:
    print(x/m)
    x=(a*x) %m