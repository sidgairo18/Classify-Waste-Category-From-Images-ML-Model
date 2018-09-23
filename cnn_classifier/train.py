import cv2
import numpy as np
import tensorflow as tf
import time
import dataset
import math
import random
from datetime import timedelta
import pdb

def get_accuracy(pred_labs, pred, true, k):

    return


np.random.seed(1)
tf.set_random_seed(2)

batch_size = 32
#batch_size = 128

classes = ['dry_other', 'dry_cans', 'dry_plastic', 'dry_paper', 'wet']
num_classes = len(classes)

#Here we take 20% of the data for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'training_data'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Reading of input data is complete. Printing a snippet of the data")
print("Number of files in Training-Set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-Set:\t\t{}".format(len(data.valid.labels)))

#pdb.set_trace()

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

#labels at the output layer
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#Network graph parameters

filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters, nam):

    #weights and biases which will be update during training
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = tf.nn.relu(layer, name = nam)

    return layer

def create_flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1, nam = 'layer_conv1')

layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2, nam = 'layer_conv2')

layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3, nam = 'layer_conv3')

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
#y_pred = tf.nn.sigmoid(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
#y_pred_cls = tf.round(y_pred)

session.run(tf.global_variables_initializer())
#session.run(tf.initialize_all_variables())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#correct_prediction = tf.equal(y_pred_cls, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
#session.run(tf.initialize_all_variables())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    

    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuray:{1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):

    global total_iterations
    global batch_size

    for i in range(total_iterations, total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, valid_batch_names, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch, y_true:y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true:y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './my-model')
	    if epoch == 200:
	        break

    total_iterations += num_iteration

train(num_iteration=12000)

print("Code Finished Running Successfully")
