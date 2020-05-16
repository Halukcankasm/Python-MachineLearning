from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


import scipy.ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

x = tf.placeholder(tf.float32, [None, 784])

"""
tf.placeholder -> yer tutucu görevini görüyor.
Tensorflow a çalıştırmasını istediğimiz x değeri
  Yani x e bir değer vereceğiz ve tf bunu hesaplayacak
"""

W = tf.Variable(tf.zeros([784,10]))
"""
Ağırlık (W) ,
 tf.Variable -> diyerek tenserflowda bir değişken tanımlıyorum
 Ağırlıklar değişken .
 784 Satır, 10 sütün
"""

b = tf.Variable(tf.zeros([10]))
""" Base , 3.Değişken, 10 Satır """


y = tf.nn.softmax(tf.matmul(x,W) + b)
"""
tf.matmul(x,W)=x.W
tf.nn.softmax() -> İçin aldığı değeri normalleştiriyor (0-1) sınırlandırıyor.
y -> Hücremin noron değeri
"""

y_ = tf.placeholder(tf.float32,[None , 10])



cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))    

"""
cross_entropy -> Düzensizlik
tf.reduce_mean -> Ortalamasını Düşürmeye çalış
tf.reduce_sum -> Toplamını düşürmeye çalış
y_=Gerçek hücre değerlerim
(y_*tf.log(y), reduction_indices=[1]) -> Hata Fonksiyonum
(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]) -> Bu hata fonk. Toplamının en az olmasını istiyorum

"""


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
train_step -> Adım adım verilerimi eğit
train:Eğit
 tf.train.GradientDescentOptimizer(0.5) ->GradientDescentOptimizer a göre eğit
 Kademeli düşürme il benim en uygun W ve b değerlerimi bul
"""


sess = tf.InteractiveSession() # Artık belli bir işleme başladığımızı gösteriyor

tf.global_variables_initializer().run()
"""
Benim değişkenlerimi başlat ve çalıştır
"""

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_xs, y_:batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#Doğruluk oranım

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

cizim = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(pyplot.imread("alti.png")))
#cizim = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("dad.png", flatten=True)))


sonuc = sess.run(tf.argmax(y,1),feed_dict={x:[cizim]})

print(sonuc)








































