from keras.utils import to_categorical
from model import GlimpseNetwork, RecurrentNetwork, ClassificationNetwork, EmissionNetwork
from dataset import load
import tensorflow as tf
import numpy as np

batch_index = 0     # to help getting next batch

def next_batch(imgs, locs, labels, batch_size=32):
    global batch_index
    """
    if batch_index + batch_size >= np.shape(imgs)[0]:
        pass
    """
    labels = to_categorical(locs, 10)
    return imgs[batch_index:batch_index+32, :, :, :], \
            locs[batch_index:batch_index+32, :], \
            labels[batch_index:batch_index+32, :]

if __name__ == '__main__':
    # Load data & placeholder
    train_features, train_labels, train_location = load()
    feature_ph = tf.placeholder(tf.float32, [None, 100, 100, 1])
    location_ph = tf.placeholder(tf.float32, [None, 6])
    label_ph = tf.placeholder(tf.int32, [None, 10])

    # Construct whole model
    gl_net = GlimpseNetwork(feature_ph)
    recurrent_net = RecurrentNetwork()
    classification_net = ClassificationNetwork()
    emission_net = EmissionNetwork()
    g_s = gl_net(location_ph)
    lstm_class_out, lstm_loc_out = recurrent_net(g_s)
    class_predict = classification_net(lstm_class_out)
    action_predict = emission_net(lstm_loc_out)

    # Compute loss
    loss_where = tf.matmul(tf.square(tf.subtract(location_ph, action_predict)), 
        tf.convert_to_tensor([[1], [0.5], [1], [0.5], [1], [1]]))
    loss_what = tf.nn.softmax_cross_entropy_with_logits(labels=label_ph, logits=class_predict)
    loss_sum = tf.reduce_mean(loss_where + loss_what)

    # Optimize
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
    var_list = tf.trainable_variables()
    grad = tf.gradients(loss_sum, var_list)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.apply_gradients(zip(grad, var_list), global_step=global_step)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            images_batch, location_batch, label_batch = next_batch(train_features, train_location, train_labels)
            _loss_value, _ = sess.run([loss_sum, train_op], feed_dict={
                feature_ph: images_batch,
                location_ph: location_batch,
                label_ph: label_batch
            })
            print('iter: ', i, '\tloss: ', _loss_value)