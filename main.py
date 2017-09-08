from model import GlimpseNetwork, RecurrentNetwork, ClassificationNetwork, EmissionNetwork
from dataset import load
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Load data & placeholder
    train_features, tain_labels, train_location = load()
    feature_ph = tf.placeholder(tf.float32, [None, 100, 100, 1])
    location_ph = tf.placeholder(tf.float32, [None, 6])
    label_ph = tf.placeholder(tf.int32, [None, 10])

    # Construct whole model
    