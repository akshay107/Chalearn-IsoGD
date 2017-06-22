import numpy as np
np.random.seed(100)
import keras

def model_main(rgb):
	import keras.backend as K
	from keras.models import Sequential
	from keras.layers.core import Dense,Activation
	from keras.layers import Convolution3D,MaxPooling3D,TimeDistributed,Flatten,ZeroPadding3D,Dropout,Lambda
	nb_classes = 249
	sum_dim1 = Lambda(lambda xin: K.sum(xin, axis = 1), output_shape=(1024,))
	model = Sequential()
	# 1st layer group
	if rgb:
		model.add(TimeDistributed(Convolution3D(16, 3, 3, 3, activation='relu',
		                    border_mode='same', name='conv1',
		                    subsample=(1, 1, 1)),
		                    input_shape=(None,3,8, 112, 112)))
	else:
		model.add(TimeDistributed(Convolution3D(16, 3, 3, 3, activation='relu',
		                    border_mode='same', name='conv1',
		                    subsample=(1, 1, 1)),
		                    input_shape=(None,1,8, 112, 112)))

	model.add(TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool1')))
	# 2nd layer group
	model.add(TimeDistributed(Convolution3D(16, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv2',
                        subsample=(1, 1, 1))))

	model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool2')))
	# 3rd layer group
	model.add(TimeDistributed(Convolution3D(32, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3b',
                        subsample=(1, 1, 1))))

	model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool3')))
	# 4th layer group
	model.add(TimeDistributed(Convolution3D(64, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv4b',
                        subsample=(1, 1, 1))))

	model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool4')))

	model.add(TimeDistributed(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding')))
	model.add(TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool5')))
	model.add(TimeDistributed(Flatten(name='flatten')))
	# FC layers group
	model.add(sum_dim1)
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	return model

def weight_sharing(rgb):
	from keras.optimizers import SGD
	model = model_main(rgb)
	model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
	return model
