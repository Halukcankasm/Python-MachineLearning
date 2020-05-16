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
 



"""

for z in range(25):
    z=2*z+1
    print("En yakın",z,"komşu kullandığımızda tutarlılık oranımız")
    
    tahmin = KNeighborsClassifier(n_neighbors=z,weights='uniform',algorithm='auto',leaf_size=30,
                              p=2,metric='euclidean',metric_params=None,n_jobs=1)
    tahmin.fit(x,y)
    y_tahmin=tahmin.predict(x) # y_tahmini x lere göre tahmin et
    basari=accuracy_score(y,y_tahmin,normalize=True,sample_weight=None)
    print(basari)

Buradan sonuçla en yakın doğruluk oranına sahip sayım 3

"""


tahmin = KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto',leaf_size=30,
                              p=2,metric='euclidean',metric_params=None,n_jobs=1)

"""
n_neighbors=3 ->En yakın komşu sayısı
weights='uniform' ->Ağırlığımız yok  
algorithm='auto' ->Algoritmamızı otamatik kendi belirlesin
metric='euclidean' -> Aradaki mesafenin karesini alıyor
"""
tahmin.fit(x,y)
y_tahmin=tahmin.predict(x) # y_tahmini x lere göre tahmin et


basari=accuracy_score(y,y_tahmin,normalize=True,sample_weight=None)
print(basari)

a=np.array([2,9,3,5,3,6,1,8,5]).reshape(1,-1)
print(tahmin.predict(a))































