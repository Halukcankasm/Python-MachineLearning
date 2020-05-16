import numpy as np
import pandas as pd


from sklearn.impute import SimpleImputer as Imputer #Soru işareti olan değerleri görmek için(alınmayan veriler)
#from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier#En yakın komşular algoritması
from sklearn.metrics import accuracy_score# % olarak veriler ile ne kadar ortüştüğünü anlayacağız
from sklearn.model_selection import train_test_split
from sklearn import cross_decomposition as cross_validation

from sklearn.datasets import load_iris
from sklearn import tree
#import pydocplus
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import sklearn

iris = datasets.load_iris()

model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)
a=np.array([6.7,3.3,5.7,2.5]).reshape(1,-1)
model.predict(a)