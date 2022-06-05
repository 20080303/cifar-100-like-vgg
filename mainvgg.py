from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

num_classes = 100 
batch_size = 128
learn=0.001
ep=80






(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = tf.squeeze(y_train,axis=1) 
(x_new,y_new) = (x_train,y_train)
for i in range(0,100):
    xx_image=tf.image.random_flip_left_right(x_train[i])
    xx_image=tf.image.random_brightness(xx_image, max_delta=0.7)
    xx_image=tf.image.random_contrast(xx_image, lower=0.2, upper=1.8)
    x_new=np.insert(x_new,0,xx_image,axis=0)
    y_new=np.insert(y_new,0,y_train[i],axis=0)
print('---------------------------------------------------------')  
(x_train,y_train) = (x_new,y_new)


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


opt = keras.optimizers.RMSprop(learning_rate=learn, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=ep,
              validation_data=(x_test, y_test),
              shuffle=True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
