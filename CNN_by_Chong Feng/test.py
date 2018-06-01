import glob
import numpy as np
import tensorflow as tf
from PIL import Image


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))


def layer(input, num_input_channels, filter_size, num_filters, training,
          use_bn=True, use_relu=True, use_pool=True):
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

    return layer


def flat_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = weight([num_inputs, num_outputs])
    biases = bias(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


filter_size1 = 3
num_filters1 = 32
filter_size2 = 3
num_filters2 = 64
filter_size3 = 3
num_filters3 = 128
filter_size4 = 3
num_filters4 = 256
fc_size = 128
num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
classes = ['maze', 'pig']
num_classes = len(classes)

test_x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='test_x')
test_x_image = tf.reshape(test_x, [-1, img_size, img_size, num_channels])
test_y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                             name='test_y_true')
test_y_true_cls = tf.argmax(test_y_true, dimension=1)

test_layer1 = layer(input=test_x_image, num_input_channels=num_channels,
                    filter_size=filter_size1, num_filters=num_filters1,
                    training=False)
test_layer2 = layer(input=test_layer1, num_input_channels=num_filters1,
                    filter_size=filter_size2, num_filters=num_filters2,
                    training=False)
test_layer3 = layer(input=test_layer2, num_input_channels=num_filters2,
                    filter_size=filter_size3, num_filters=num_filters3,
                    training=False)
# test_layer4 = layer(input=test_layer3, num_input_channels=num_filters3,
#                     filter_size=filter_size4, num_filters=num_filters4,
#                     training=False)
test_layer_flat, test_num_features = flat_layer(test_layer3)
test_layer_fc1 = fc_layer(input=test_layer_flat, num_inputs=test_num_features,
                          num_outputs=fc_size, use_relu=True)
test_layer_fc2 = fc_layer(input=test_layer_fc1, num_inputs=fc_size,
                          num_outputs=num_classes, use_relu=False)

test_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=test_layer_fc2, labels=test_y_true)
test_y_pred_cls = tf.argmax(test_layer_fc2, dimension=1)

test_correct_prediction = tf.equal(test_y_pred_cls, test_y_true_cls)
test_loss = tf.reduce_mean(test_cross_entropy)
test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

test_path = "data/original/*.png"

test_set = []
test_files = glob.glob(test_path)
for file in test_files:
    img = Image.open(file)
    test_set.append((np.array(img)))
    img.close()

test_labels = []
file_test = "data/original/test_label.txt"
ft = open(file_test, "r")
try:
    for line in ft:
        test_labels.append(line.strip("\n"))
finally:
    ft.close()

test_label = np.zeros((len(test_labels), num_classes))
for i in range(len(test_labels)):
    test_label[i, int(test_labels[i])] = 1

sess = tf.Session()
saver = tf.train.Saver()
if tf.train.latest_checkpoint('ckpts') is not None:
    saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
else:
    assert 'can not find checkpoint folder path!'

test = np.array(test_set).reshape(len(test_set), img_size_flat)
test_los, test_acc = sess.run(
    [test_loss, test_accuracy],
    feed_dict={test_x: test, test_y_true: test_label})
msg = "Test Accuracy: {0:>6.1%}, Loss: {1:.3f}"
print(msg.format(test_acc, test_los))

sess.close()
