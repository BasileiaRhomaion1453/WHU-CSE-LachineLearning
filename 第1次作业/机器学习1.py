import matplotlib.pyplot as plt
import numpy as np
xi = [0.95,3,4,5.07,6.03,8.21,8.85,12.02,15]
yi = [5.1,8.7,11.5,13,15.3,18,21,26.87,32.5]

xv1=1/9*sum(xi)
yv1=1/9*sum(yi)

k1=(sum(np.multiply(xi,yi))-9*xv1*yv1)/(sum(np.multiply(xi,xi))-9*xv1*xv1)
b1=yv1-k1*xv1

x1 = np.linspace(0, 15, 50)
y1 = k1*x1 + b1

epsilon=0.001#迭代阈值
alpha=0.0001#学习率
b2=0
k2=0
#损失函数
error0=0
error1=0
cnt=0
while True:
    cnt=cnt+1
    for i in np.arange(9):
        cancha=b2+k2*xi[i]-yi[i]
        b2=b2-alpha*cancha
        k2=k2-alpha*cancha*xi[i]
    error1=0    
    for i in np.arange(9):
        error1=error1+(b2+k2*xi[i]-yi[i])**2*0.5
    c=error1-error0  
    if abs(c)<epsilon:
        break
    else:
        error0=error1
x2 = np.linspace(0, 15, 50)
y2 = b2+k2*x2
plt.subplot(2, 1, 1)
plt.plot(x1, y1,color="red")
plt.scatter(xi, yi,color="green")
plt.subplot(2, 1, 2)
plt.plot(x2, y2,color="red")
plt.scatter(xi, yi,color="blue")
plt.show()