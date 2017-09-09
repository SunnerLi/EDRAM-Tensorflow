import tensorlayer as tl
import tensorflow as tf
import numpy as np

def toL(_tensor):
    """
        Casting the tensor (tensorflow object) into Layer (tensorlayer object)

        Arg:    _tensor - The tensor object
        Ret:    The layer object with no other computation
    """
    input_layer_index = -1
    while True:
        try:
            input_layer_index += 1
            return tl.layers.InputLayer(_tensor, name='input_layer'+str(input_layer_index))
        except Exception:
            continue

class GlimpseNetwork(object):
    def __init__(self, image_ph):
        self.image_ph = image_ph

    def __call__(self, _action):
        # ------------------------------
        #  Construct the first part
        # ------------------------------
        # Spatial transformation
        self.network = tl.layers.SpatialTransformer2dAffineLayer(toL(self.image_ph), toL(_action))

        # Conv1
        self.network = tl.layers.Conv2dLayer(self.network, shape = [3, 3, 1, 16], name='glimpse_conv1')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv1_bn')
        self.network = tf.nn.relu(self.network.outputs)

        # Conv2 + pooling
        self.network = tl.layers.Conv2dLayer(toL(self.network), shape = [3, 3, 16, 32], name='glimpse_conv2')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv2_bn')
        self.network = tf.nn.relu(self.network.outputs)
        self.network = tl.layers.MaxPool2d(toL(self.network), name='maxpool1')

        # Conv3
        self.network = tl.layers.Conv2dLayer(self.network, shape = [3, 3, 32, 64], name='glimpse_conv3')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv3_bn')
        self.network = tf.nn.relu(self.network.outputs)

        # Conv4 + pooling
        self.network = tl.layers.Conv2dLayer(toL(self.network), shape = [3, 3, 64, 128], name='glimpse_conv4')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv4_bn')
        self.network = tf.nn.relu(self.network.outputs)
        self.network = tl.layers.MaxPool2d(toL(self.network), name='maxpool2')

        # Conv5
        self.network = tl.layers.Conv2dLayer(self.network, shape = [3, 3, 128, 160], name='glimpse_conv5')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv5_bn')
        self.network = tf.nn.relu(self.network.outputs)

        # Conv6 + pooling
        self.network = tl.layers.Conv2dLayer(toL(self.network), shape = [3, 3, 160, 192], name='glimpse_conv6')
        self.network = tl.layers.BatchNormLayer(self.network, name='glimpse_conv6_bn')
        self.network = tf.nn.relu(self.network.outputs)
        self.network = tl.layers.MaxPool2d(toL(self.network), name='maxpool3')

        # ------------------------------
        #  Construct the second part
        # ------------------------------
        # conv result
        self.network_conv_part = tl.layers.FlattenLayer(self.network)
        self.network_conv_part = tl.layers.DenseLayer(self.network_conv_part, n_units = 1024, name='glimpse_fc1')
        self.network_conv_part = tl.layers.BatchNormLayer(self.network_conv_part, name='glimpse_fc1_bn')
        self.network_conv_part = tf.nn.relu(self.network_conv_part.outputs)

        # Location result
        self.network_loc_part = tl.layers.DenseLayer(toL(_action), n_units = 1024, name='glimpse_fc2')
        self.network_loc_part = tl.layers.BatchNormLayer(self.network_loc_part, name='glimpse_fc2_bn')
        self.network_loc_part = tf.nn.relu(self.network_loc_part.outputs)

        # Combine
        self.network = self.network_conv_part * self.network_loc_part
        return self.network

class RecurrentNetwork(object):
    def __call__(self, glimpse_result):
        # Classification part RNN
        self.classification_part = tl.layers.DenseLayer(toL(glimpse_result), n_units = 2048, name='core_fc1')
        self.classification_part = toL(tf.expand_dims(self.classification_part.outputs, axis=-1))
        self.classification_part = tl.layers.RNNLayer(self.classification_part, cell_fn = tf.nn.rnn_cell.LSTMCell, n_hidden = 512, n_steps = 1, name='classification_rnn')

        # Emission part RNN
        self.emission_part = tf.reshape(self.classification_part.outputs, [-1, 512])
        self.emission_part = tl.layers.DenseLayer(toL(self.emission_part), n_units = 2048, name ='core_fc2')
        self.emission_part = toL(tf.expand_dims(self.emission_part.outputs, axis=-1))
        self.emission_part = tl.layers.RNNLayer(self.emission_part, cell_fn = tf.nn.rnn_cell.LSTMCell, n_hidden = 512, n_steps = 1, name='emission_rnn')
        return self.classification_part.outputs, self.emission_part.outputs

class ClassificationNetwork(object):
    def __call__(self, lstm_result):
        self.network = tf.reshape(lstm_result, [-1, tf.cast(lstm_result.shape[1] * lstm_result.shape[2], tf.int32)])
        self.network = tl.layers.DenseLayer(toL(self.network), n_units = 1024, name='classification_fc1')
        self.network = tl.layers.BatchNormLayer(self.network, name='classification_fc1_bn')
        self.network = tf.nn.relu(self.network.outputs)
        self.network = tl.layers.DenseLayer(toL(self.network), n_units = 1024, name='classification_fc2')
        self.network = tl.layers.BatchNormLayer(self.network, name='classification_fc2_bn')
        self.network = tf.nn.relu(self.network.outputs)
        self.network = tl.layers.DenseLayer(toL(self.network), n_units = 10, act = tf.nn.softmax, name='classification_fc3')
        return self.network.outputs

class EmissionNetwork(object):
    def __call__(self, lstm_result):
        self.network = tf.reshape(lstm_result, [-1, tf.cast(lstm_result.shape[1] * lstm_result.shape[2], tf.int32)])
        self.network = tl.layers.DenseLayer(toL(self.network), n_units = 1024, act = tf.nn.tanh, name='emission_fc')
        self.network = self.network.outputs
        self.network = tf.stack([
            tf.clip_by_value(self.network[:, 0], 0.0, 1.0), self.network[:, 1], self.network[:, 2],
            self.network[:, 3], tf.clip_by_value(self.network[:, 4], 0.0, 1.0), self.network[:, 5]
        ], axis=1)
        return self.network