import einops
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class Conv3Plus1D(keras.layers.Layer):
  def __init__(self, filters, padding):
    """
      A sequence of convolutional layers that apply the convolution operation over the
      spatial dimensions
    """
    super().__init__()
    self.seq = keras.Sequential([
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(3, 3, 3),
                      padding=padding)
        ])

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'seq': self.seq
      })
      return config

  def call(self, x):
    return self.seq(x)

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters):
    super().__init__()
    self.seq = keras.Sequential([
        Conv3Plus1D(filters=filters,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv3Plus1D(filters=filters,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'seq': self.seq
      })
      return config

  def call(self, x):
    return self.seq(x)

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'seq': self.seq
      })
      return config

  def call(self, x):
    return self.seq(x)

def add_residual_block(input, filters):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.experimental.preprocessing.Resizing(self.height, self.width)

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'height': self.height,
          'width': self.width,
          'resizing_layer': self.resizing_layer,
      })
      return config

  def call(self, video):
    """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height,
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos


def spatio_network(frames, height, width, channels):
    input_shape = (None, frames, height, width, channels)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv3Plus1D(filters=16, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16)
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32)
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64)
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 128)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(input, x)

    model.compile(loss="BinaryCrossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    # opt = tf.keras.optimizers.SGD(learning_rate=0.001, decay=0.0001)
    # model.compile(optimizer=opt, loss="BinaryCrossentropy", metrics="binary_accuracy")

    return model