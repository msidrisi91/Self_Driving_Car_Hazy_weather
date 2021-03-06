import numpy as np
from keras.preprocessing import image
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
import dehaze
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

adam = Adam(0.0001)
L2NormConst = 0.001
model = Sequential()
model.add(Conv2D(24, [5, 5], activation='relu', input_shape=[66, 200, 3], strides=[2, 2], kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(36, [5, 5], activation='relu', strides=[2,2], kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(48, [5, 5], activation='relu', strides=[2,2], kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, [3, 3], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, [3, 3], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.summary()

model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.load_weights('./zweights-0.0696.h5')

img_str = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img_str.shape

smoothed_angle = 0
path = "fogg/"

i = 42500

while(cv2.waitKey(10) != ord('q') and i<45000):
	if i == 44215: i = 44800
	img1 = cv2.imread(path + str(i) + ".jpg", 1)
	# img1 = image.load_img(path + str(i) + ".jpg",color_mode='rgb')
	cv2.imshow("frame", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
	img = dehaze.dehaze(img1)[..., ::-1].astype(np.float32)

	img1 = image.image.img_to_array(img) / 255.0
	cv2.imshow("dehazed", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

	# img = cv2.resize(img1, (66, 200))[..., ::-1].astype(np.float32)
	img = image.load_img(path + str(i) + ".jpg", color_mode='rgb', target_size=[66, 200])
	img = image.img_to_array(img)/255.0
	img_resh = np.reshape(img, [1, 66, 200, 3])

	degrees = model.predict(img_resh) * 180.0 / 3.14 + 8

	print("Predicted steering angle: " + str(degrees) + " degrees")
	smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
	M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
	dst = cv2.warpAffine(img_str, M, (cols, rows))
	cv2.imshow("steering wheel", dst)
	i += 1

cv2.destroyAllWindows()
