import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data = pd.read_csv("linear.csv")

x_data=data["metrekare"]#metrekarenin altındaki tüm verileri aldık
y_data=data["fiyat"]#fiyatın altındaki tüm verileri aldık

x=np.array(x_data)
y=np.array(y_data)

x=x.reshape(99,1)
y=y.reshape(99,1)


lineerregresyon = lr()

lineerregresyon.fit(x, y)#x ve ye değerlerini uygun olarak yerlestir

lineerregresyon.predict(x)#Tahmin et fonk. 
"""
x eksenine göre tahmin yapacağız

Doğrumuzun formulü = m(eğim).x + b(kesiştiği nokta)

"""

print("Eğim(m):",lineerregresyon.coef_)
print("Y ekseninde kesiştiği nok(b):",lineerregresyon.intercept_)

m=lineerregresyon.coef_
b=lineerregresyon.intercept_

a=np.arange(150)#0-149 a kadar bir matrix oluşturur
a=a.reshape(150,1)

plt.scatter(x, y)
plt.plot(a,m*a+b)#Doğru(plot(x ekseni,y ekseni)) #TAHMİN DOĞRUMUZ


# z=float(input("Kaç metrekare"))
# tahmin =m*z+b
# print(tahmin)

# plt.scatter(z, tahmin, c="red",marker=">")
# plt.show()
# print("y=",m,"x+",b)


for i in range(len(y)):#Y nin içindeki her sayı değeri için
    hatakaresi_lineer =+ (y[i]-(m*x[i]+b))**2 
    
    




