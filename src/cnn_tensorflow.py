from tkinter import _Padding
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Input
from keras.layers.convolutional import MaxPooling2D
from tf.keras.layers.Conv2DTranspose import Deconv
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

# load data
#
#   (x_train, y_train), (x_test, y_test) = mnist.load_data()
#   x_train, x_test = x_train / 255.0, x_test / 255.0
#
#


# CNN

#layer 1
model = Sequential()
model.add(Conv2D(1, kernel_size=(960, 540), activation='relu', input_shape=(1280, 720, 1))) ##output (320, 180, 1)

model.add(Deconv(1, kernel_size=(960, 540), strides=(1, 1), activation='relu')) ##output (1280, 720, 32)


#layer 4
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), strides= (1, 1), padding = 'same', activation='relu', input_shape=(1280, 720, 1))) ##output (1280, 720, 64)
model.add(Conv2D(64, kernel_size=(3, 3), strides= (1, 1), padding = 'same', activation='relu', input_shape=(1280, 720, 64))) ##output (1280, 720, 64)
model.add(Conv2D(32, kernel_size=(3, 3), strides= (1, 1), padding = 'same', activation='relu', input_shape=(1280, 720, 64))) ##output (1280, 720, 32)
model.add(Conv2D(16, kernel_size=(3, 3), strides= (1, 1), padding = 'same', activation='relu', input_shape=(1280, 720, 32))) ##output (1280, 720, 16)

model.add(Deconv(32, kernel_size=(3, 3), strides=(1, 1), padding = 'same', activation='relu')) ##output (1280, 720, 32)
model.add(Deconv(64, kernel_size=(3, 3), strides=(1, 1), padding = 'same', activation='relu')) ##output (1280, 720, 64)
model.add(Deconv(64, kernel_size=(3, 3), strides=(1, 1), padding = 'same', activation='relu')) ##output (1280, 720, 64)
model.add(Deconv(1, kernel_size=(3, 3), strides=(1, 1), padding = 'same', activation='relu'), bias = 0.1) ##output (1280, 720, 1)



#layer 정보 출력
model.summary()

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

#학습
model.fit(x_train, y_train, epochs=5, batch_size=200, verbose=2)
#테스트
model.evaluate(x_test,  y_test, verbose=2)






#Original Code
# model = Sequential()
# model.add(Conv2D(12, kernel_size=(5, 5), activation='relu', input_shape=(120, 60, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(20, kernel_size=(4, 4), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)



