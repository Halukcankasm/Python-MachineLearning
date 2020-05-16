import numpy as np
import pandas as pd


from sklearn.impute import SimpleImputer as Imputer #Soru işareti olan değerleri görmek için(alınmayan veriler)
#from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier#En yakın komşular algoritması
from sklearn.metrics import accuracy_score# % olarak veriler ile ne kadar ortüştüğünü anlayacağız
from sklearn.model_selection import train_test_split
from sklearn import cross_decomposition as cross_validation


veri = pd.read_csv("breast-cancer-wisconsin.data")

"""Veilerimizde '?' işareti olan yerler için ne yapmamız gerekiyor ->>"""

veri.replace('?',-99999, inplace=True)#soru işareti olan yerleri -99999 ile güncelledik
veri = veri.drop(['id'],axis=1)#İd stününü sildik


y = np.array(veri.benormal)
x=np.array(veri.drop(['benormal'],axis=1))

imp = Imputer(missing_values=-99999,strategy="mean")
 #skitlearn bilmediğimiz bu verilerin ortalamasını alarak tahminimizin az etkilenmesini sağlıyor
x=imp.fit_transform(x)#Tekrardan x e döndürüyoruz
 

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
"""
Verimizi test ve eğitim kümesi olarak ayırıyorduk,test_size=0.33 belirleyerek
verimizin 1/3 ni test olarak geriye kalan kısmınıda eğtim kümesi olarak akıyoruz
X_train -> x deki eğitim kümem
X_test -> x deki test kümem
y_train -> y deki eğitim kümem
y_test -> y deki test kümem
"""

tahmin = KNeighborsClassifier()#Parantez içerisini boş bıraktığımızda kendi bir model oluşturacak
tahmin.fit(X_train,y_train)
basari=tahmin.score(X_test,y_test)

print("Yüzde",basari*100,"oranında:")
a=np.array([2,9,3,5,3,6,1,8,5]).reshape(1,-1)
print(tahmin.predict(a))
































