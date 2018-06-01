import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
# from tf_cnnvis import *


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))


def layer(input, num_input_channels, filter_size, num_filters, use_bn=False,
          use_relu=True, use_pool=True, use_dropout=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight(shape)
    biases = bias(num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],
                         padding="SAME")
    layer += biases

    if use_bn:
        layer = tf.layers.batch_normalization(layer, training=training)

    if use_relu:
        layer = tf.nn.relu(layer)

    if use_pool:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding="SAME")

    if use_dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    return layer


def save_layer(layer, image, image_name, use):
    image = image.reshape(img_size_flat)
    feed_dict = {x: [image], keep_prob: 0.5}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
    fig.savefig("data/layers/features/" + image_name +
                "_" + use + ".png")


keep_prob = tf.placeholder(tf.float32)

filter_size1 = 3
num_filters1 = 32
filter_size2 = 3
num_filters2 = 64
filter_size3 = 3
num_filters3 = 128
filter_size4 = 3
num_filters4 = 256
num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)

training = True

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

layer1 = layer(input=x_image, num_input_channels=num_channels,
               filter_size=filter_size1, num_filters=num_filters1)

session = tf.Session()
session.run(tf.global_variables_initializer())

img0 = Image.open("record/images/not_preprocessed/test/test_34.png")
image0 = np.array(img0)
img1 = Image.open("record/images/not_preprocessed/test/test_31.png")
image1 = np.array(img1)

save_layer(layer=layer1, image=image0, image_name="maze", use="conv")
save_layer(layer=layer1, image=image1, image_name="pig", use="conv")

# image0 = image0.reshape(img_size_flat)
# feed_dict = {x: [image0], keep_prob: 0.5}
# layers = ["r", "p", "c"]
# is_success = deconv_visualization(sess_graph_path=session,
#                                   value_feed_dict=feed_dict,
#                                   input_tensor=x_image, layers=layers,
#                                   path_logdir="record/images/layers/maze/",
#                                   path_outdir="record/images/layers/maze/")

# image1 = image1.reshape(img_size_flat)
# feed_dict = {x: [image1], keep_prob: 0.5}
# layers = ["r", "p", "c"]
# is_success = deconv_visualization(sess_graph_path=session,
#                                   value_feed_dict=feed_dict,
#                                   input_tensor=x_image, layers=layers,
#                                   path_logdir="record/images/layers/pig/",
#                                   path_outdir="record/images/layers/pig/")

session.close()
img0.close()
img1.close()
