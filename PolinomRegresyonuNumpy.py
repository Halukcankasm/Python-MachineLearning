import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data = pd.read_csv("linear.csv")

x=data["metrekare"]#metrekarenin altındaki tüm verileri aldık
y=data["fiyat"]#fiyatın altındaki tüm verileri aldık

x=np.array(x)
y=np.array(y)

a,b,c,d=np.polyfit(x, y, 3)
#a,b,c=np.polyfit(x, y, 2)#2.dereceden türevini alarak x ve y düzlemlerii oturt
#2.dereceden bir formül için = ax^2 + bx + c , katsayılar = a,b,c

z= np.arange(150)

plt.scatter(x, y)

plt.plot(z,a*(z**3)+b*(z**2)+c*(z**1)+d)
plt.show()
#2.derecen =plt.plot(z,a*z**2+b*z+c)

# h= float(input("metrekare"))
# sonuc = a*h*h*h+b*h*h+c*h+d
# print(sonuc)

for i in range(int(len(y))):
    hatakaresi =+ ((y[i])-(a*(x[i])**3+b*(x[i]**2)+c*(x[i])+d))**2
    
