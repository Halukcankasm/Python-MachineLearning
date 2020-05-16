from numpy import exp , array , random , dot , mean , abs

#exp = e^ (e üzeri fonksiyon) demek

import numpy

girdi = array([[0,0,1],[1,1,1],[1,0,1]])

gerceksonuclar = array([[0,1,1]]).T
"""'array([[0,1,1]]).T' , T= transpozunu aldık"""
gerceksonuclar=array(gerceksonuclar).reshape(3,1)

agirlik = array([1.0,1.0,1.0]).T
#ağırlıklar rastgele olucak
agirlik=array(agirlik).reshape(3,1)


for tekrar in range(1000):
    hucredegeri = dot(girdi,agirlik)
    print("Hücre değeri :",hucredegeri)
    tahmin = 1/(1+exp(-hucredegeri))
    print("tahmin:",tahmin)
    agirlik = agirlik + dot(girdi.T,((gerceksonuclar-tahmin)*tahmin*(1- tahmin)))
    print("Ağırlık:",agirlik)
    print("hata değeri:",str(mean(abs(gerceksonuclar-tahmin))))
    #(gerceksonuclar-tahmin) = hata katsayımız
    #tamin*(1- tahmin) = sigmoid fonksiyonunun türevi
    
    
    
    
ornek_girdi=array([1,0,0])
ornek_hucre_degeri=dot(ornek_girdi,agirlik)    
    
print("Örnek problem",1/(1+exp(-ornek_hucre_degeri)))    
    
    
