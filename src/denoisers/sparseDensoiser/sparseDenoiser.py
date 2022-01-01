import pandas as pd
import numpy as np 
import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow.keras.datasets  import mnist


### IMPORT THE MNIST DATA #### 

train,test=mnist.load_data()
x_train,y_train=train
x_test,y_test=test

## THE IMAGE SHAPE OF MNSIT IS 28 X 28 ### 
print(x_train.shape)

## PLOT THE IMAGE ##
for i in range(2):
  plt.imshow(x_train[i],cmap='gray')
  plt.show()
  plt.close()
  break

### ADD NOISE TO THE DATA ### 

x_train_noisy=(x_train+np.random.normal(10,20,size=x_train.shape))*1./255
x_test_noisy=(x_test+np.random.normal(10,20,size=x_test.shape))*1./255

###  PLOT THE NOISY IMAGE ###
for i in range(5):
  plt.imshow(x_train_noisy[i],cmap='gray')
  plt.show()
  plt.close()
  break

#### DENOISER ####
inputs=tf.keras.Input((28,28,1))
enc=tf.keras.layers.Conv2D(64,5,padding='same',activation='relu')(inputs)
enc=tf.keras.layers.MaxPooling2D(strides=2)(enc)
enc=tf.keras.layers.Conv2D(32,5,padding='same',activation='relu')(enc)
enc=tf.keras.layers.MaxPooling2D(strides=2)(enc)
enc=tf.keras.layers.Conv2D(16,5,padding='same',activation='relu')(enc)


enc=tf.keras.layers.Conv2D(8,5,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.L1(1e-3))(enc)


dec=tf.keras.layers.Conv2D(16,5,padding='same',activation='relu')(enc)
dec=tf.keras.layers.UpSampling2D((2,2))(dec)
dec=tf.keras.layers.Conv2D(32,5,padding='same',activation='relu')(dec)
dec=tf.keras.layers.UpSampling2D((2,2))(dec)
dec=tf.keras.layers.Conv2D(64,5,padding='same',activation='relu')(dec)

output=tf.keras.layers.Conv2D(1,5,padding='same',activation='relu')(dec)


sparse_autoencoder=tf.keras.Model(inputs,output,name='SparseDenoiser')



tf.keras.utils.plot_model(sparse_autoencoder)

#Print Summary
sparse_autoencoder.summary()

x_train_noisy=x_train_noisy.reshape((60000,28,28,1))
x_train=x_train.reshape((60000,28,28,1))
history=sparse_autoencoder.fit(x_train_noisy,x_train,batch_size=256,validation_split=0.2,epochs=100)

plt.rcParams['figure.figsize']=(10,10)

#### PLOT HOSTORY AND CHECK OVERFITTING  ####
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.xlabel('Epochs',fontsize=25)
plt.ylabel('Loss',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()
plt.close()






### EVALUATE MODEL ###

evaluate=sparse_autoencoder.evaluate(x_test_noisy.reshape((len(x_test_noisy),28,28,1)))

print(f'The loss on test data is:\n{evaluate}')



