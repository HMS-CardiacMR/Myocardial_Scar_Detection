from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, TimeDistributed, LSTM, Dense, Input, GRU, Dropout,\
  Conv3D, MaxPool3D, BatchNormalization, Activation, Reshape, SeparableConv2D
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras

input_shape_3d = (20, 64, 64, 1)

def cnn_3d_batchnorm_LSTM(intput_shape=input_shape_3d):

  model = Sequential()
  model.add(Conv3D(32, kernel_size=3, activation='relu', padding="same", strides=1,
                   kernel_initializer='he_normal', input_shape=intput_shape, kernel_regularizer=L1L2(1e-5, 1e-5)))

  model.add(BatchNormalization())

  model.add(Conv3D(32, kernel_size=3, strides=1, activation='relu'))
  model.add(BatchNormalization())
  model.add(Conv3D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
  model.add(MaxPool3D(pool_size=2))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv3D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
  model.add(MaxPool3D(pool_size=2))
  model.add(BatchNormalization())
  model.add(Conv3D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
  model.add(MaxPool3D(pool_size=2))
  model.add(Dropout(0.4))

  model.add(Reshape((-1, 64)))
  # model.add(LSTM(32, return_sequences=True))
  model.add(SeqSelfAttention(attention_activation='tanh'))
  model.add(LSTM(10, return_sequences=False))
  model.add(Dense(1, activation='sigmoid'))

  # opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.001)
  # model.compile(optimizer=opt, loss="BinaryCrossentropy", metrics=METRICS)
  # model.compile(optimizer=opt, loss=wmsqe, metrics=["mape"])
  opt = tf.keras.optimizers.SGD(learning_rate=0.001, decay=0.0001)
  model.compile(optimizer=opt, loss="BinaryCrossentropy", metrics="binary_accuracy")
  return model