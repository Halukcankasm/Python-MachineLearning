import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lr
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as Pr

veri = pd.read_csv("2016dolaralis.csv")
print(veri)

x=veri["Gun"]
y=veri["Fiyat"]

x=np.array(x)
y=np.array(y)

x = x.reshape(251,1)
y = y.reshape(251,1)


plt.scatter(x, y)


#Linear Regresyon----------------
tahmin_lineer = Lr()
tahmin_lineer.fit(x,y)#Verileri x ve y eksenine oturtuyoruz,
tahmin_lineer.predict(x)#x(gün^e göre tahmin etmek , yani 7.günde fiyat kaç olur
#X eksenine göre Y yi tahmin edeceğiz

plt.plot(x,tahmin_lineer.predict(x),color="red")


#Polinom Regresyon-----------------------
tahmin_polinom = Pr(degree=2)#2.dereceden fonk olsun
xYeni = tahmin_polinom.fit_transform(x) #x için yeni bir matrix oluşturucağız , tahmin için oluşturduğumuz kısa form

polinom_model = Lr()
polinom_model.fit(xYeni,y)
polinom_model.predict(xYeni)

plt.plot(x,polinom_model.predict(xYeni),color="orange")

#8.derecen Polinom Regresyom----------------------------------
tahmin_polinom8 = Pr(degree=8)#8.dereceden fonk olsun
xYeni = tahmin_polinom8.fit_transform(x) #x için yeni bir matrix oluşturucağız , tahmin için oluşturduğumuz kısa form

polinom_model8 = Lr()
polinom_model8.fit(xYeni,y)
polinom_model8.predict(xYeni)

plt.plot(x,polinom_model8.predict(xYeni),color="black")

plt.show()

print((float(y[201]) - float(polinom_model.predict(xYeni)[201]))**2)
#◘200.gündeki dolar değeri


# #3.derecen Polinom Regresyom
# tahmin_polinom3 = Pr(degree=3)#3.dereceden fonk olsun
# xYeni3 = tahmin_polinom3.fit_transform(x) #x için yeni bir matrix oluşturucağız , tahmin için oluşturduğumuz kısa form

# polinom_model3 = Lr()
# polinom_model3.fit(xYeni3,y)
# polinom_model3.predict(xYeni3)

# plt.plot(x,polinom_model3.predict(xYeni3),color="blue")




hatakaresi_lineer = 0
hatakaresi_polinom = 0

for i in range(len(xYeni)):
    hatakaresi_polinom = hatakaresi_polinom + (float(y[i]) - float(polinom_model.predict(xYeni)[i]))**2
    
    
for i in range(len(y)):
    hatakaresi_lineer = hatakaresi_lineer + (float(y[i]) - float(tahmin_lineer.predict(x)[i]))**2  


hatakaresi_polinom = 0
enkucuk_hata = 5
for a in range (150):
    
    tahmin_polinom = Pr(degree=a+1)
    xYeni = tahmin_polinom.fit_transform(x) #x için yeni bir matrix oluşturucağız , tahmin için oluşturduğumuz kısa form

    polinom_model = Lr()
    polinom_model.fit(xYeni,y)
    polinom_model.predict(xYeni)
    
    for i in range(len(xYeni)):
        hatakaresi_polinom = hatakaresi_polinom + (float(y[i]) - float(polinom_model.predict(xYeni)[i]))**2
    print(a+1,".dereceden fonksiyondaki hata payı,",hatakaresi_polinom)
    if enkucuk_hata>hatakaresi_polinom:
        enkucuk_hata=hatakaresi_polinom
    hatakaresi_polinom = 0
print(enkucuk_hata)





















