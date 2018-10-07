# import the necessary packages
import keras
from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import regularizers

class DeepSeaNet:
	@staticmethod
	def build(width, height, depth, classes,emb,dropout_value):
		cnn_inputShape =Input(shape=(height,width,depth))
		cnn = Conv2D(32, (5, 5), padding="same",activation='relu') (cnn_inputShape)
		cnn = MaxPooling2D(pool_size=(3, 3), strides=(3, 3)) (cnn)
		cnn = Conv2D(50, (3, 3), padding="same",activation='relu') (cnn)
		cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (cnn)
		cnn_outputShape = Flatten()(cnn)
		cnn_model = Model(cnn_inputShape,cnn_outputShape)

		dnn_inputShape = Input(shape=(1,))
		dnn = Dense(1)(dnn_inputShape)
		dnn_outputShape = Dense(1,activation='relu')(dnn)
		dnn_model = Model(dnn_inputShape,dnn_outputShape)
		if (emb == 'True'):
			mix = concatenate([cnn_outputShape,dnn_outputShape])
			mix = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01))(mix)
			mix = Dropout(dropout_value)(mix)
			mix_outputShape = Dense(classes,activation='softmax')(mix)
			mixed_model = Model([cnn_inputShape,dnn_inputShape],mix_outputShape)
		else:
			mix = Dense(512, activation='relu')(cnn_outputShape)
			mix_outputShape = Dense(classes, activation='softmax')(mix)
			mixed_model = Model(cnn_inputShape, mix_outputShape)
		# return the constructed network architecture
		return mixed_model