import glob
import numpy as np
import tensorflow as tf
import time
import os
from datetime import timedelta
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
batch_size = 32
ITERATION_NUMBER = 100

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer1 = layer(input=x_image, num_input_channels=num_channels,
               filter_size=filter_size1, num_filters=num_filters1,
               training=True)
layer2 = layer(input=layer1, num_input_channels=num_filters1,
               filter_size=filter_size2, num_filters=num_filters2,
               training=True)
layer3 = layer(input=layer2, num_input_channels=num_filters2,
               filter_size=filter_size3, num_filters=num_filters3,
               training=True)
# layer4 = layer(input=layer3, num_input_channels=num_filters3,
#                filter_size=filter_size4, num_filters=num_filters4,
#                training=True)
layer_flat, num_features = flat_layer(layer3)
layer_fc1 = fc_layer(input=layer_flat, num_inputs=num_features,
                     num_outputs=fc_size, use_relu=True)
layer_fc2 = fc_layer(input=layer_fc1, num_inputs=fc_size,
                     num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


training_path = "data/augmentation/split/training/*.png"

training_set = []
training_files = glob.glob(training_path)
training_files = training_files
for file in training_files:
    img = Image.open(file)
    training_set.append((np.array(img)))
    img.close()

training_labels = []
file_training = "data/augmentation/split/training_label.txt"
ftr = open(file_training, "r")
try:
    for line in ftr:
        training_labels.append(line.strip("\n"))
finally:
    ftr.close()

training_labels = training_labels
training_label = np.zeros((len(training_labels), num_classes))
for i in range(len(training_labels)):
    training_label[i, int(training_labels[i])] = 1

test_path = "data/augmentation/split/test/*.png"

test_set = []
test_files = glob.glob(test_path)
test_files = test_files
for file in test_files:
    img = Image.open(file)
    test_set.append((np.array(img)))
    img.close()

test_labels = []
file_test = "data/augmentation/split/test_label.txt"
ft = open(file_test, "r")
try:
    for line in ft:
        test_labels.append(line.strip("\n"))
finally:
    ft.close()
test_labels = test_labels
test_label = np.zeros((len(test_labels), num_classes))
for i in range(len(test_labels)):
    test_label[i, int(test_labels[i])] = 1


sess = tf.Session()
sess.run(tf.global_variables_initializer())

var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
if tf.train.latest_checkpoint('ckpts') is not None:
    saver.restore(sess, tf.train.latest_checkpoint('ckpts'))

start_time = time.time()
epoch = 0
for i in range(ITERATION_NUMBER):
    training_batch = zip(range(0, len(training_set), batch_size),
                         range(batch_size, len(training_set) + 1, batch_size))
    for start, end in training_batch:
        x_batch = np.array(
            training_set[start: end]).reshape(batch_size, img_size_flat)
        y_batch = training_label[start: end]
        feed_dict = {x: x_batch, y_true: y_batch}
        sess.run(train_op, feed_dict=feed_dict)

    tr_loss = []
    tr_acc = []
    for start, end in training_batch:
        x_batch_tr = np.array(
            training_set[start: end]).reshape(batch_size, img_size_flat)
        y_batch_tr = training_label[start: end]
        feed_dict_tr = {x: x_batch_tr, y_true: y_batch_tr}
        tr_loss.append(sess.run(loss, feed_dict=feed_dict_tr))
        tr_acc.append(sess.run(accuracy, feed_dict=feed_dict_tr))

    test_batch = zip(range(0, len(test_set), batch_size),
                     range(batch_size, len(test_set) + 1, batch_size))
    test_loss = []
    test_acc = []
    for start, end in test_batch:
        x_batch_test = np.array(
            test_set[start: end]).reshape(batch_size, img_size_flat)
        y_batch_test = test_label[start: end]
        feed_dict_test = {x: x_batch_test, y_true: y_batch_test}
        test_loss.append(sess.run(loss, feed_dict=feed_dict_test))
        test_acc.append(sess.run(accuracy, feed_dict=feed_dict_test))

    msg = "Epoch {0} --- Train Accuracy: {1:>6.1%}, Loss: {2:.3f}, \
                     Test Accuracy: {3:>6.1%}, Loss: {4:.3f}"
    print(msg.format(epoch + 1, np.mean(tr_acc), np.mean(tr_loss),
                     np.mean(test_acc), np.mean(test_loss)))
    epoch += 1
    save_path = os.path.join('ckpts', 'cnn.ckpt')
    saver.save(sess, save_path)
end_time = time.time()
time_dif = end_time - start_time
print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


sess.close()
