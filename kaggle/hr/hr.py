import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_fetures(record):
    default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [""], [""]]
    atisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, left, promotion_last_5years, sales, salary = tf.decode_csv(
        record, default)
    print(math.pi)
    return tf.stack([atisfaction_level, last_evaluation, (2 * tf.atan(number_project) / math.pi),
                     (2 * tf.atan(average_montly_hours) / math.pi), (2 * tf.atan(time_spend_company) / math.pi),
                     work_accident, promotion_last_5years])


def parse_left_label(record):
    default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [""], [""]]
    atisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, left, promotion_last_5years, sales, salary = tf.decode_csv(
        record, default)
    return tf.stack([left])


def loadDataset(filename, header: bool = True, minibatch_size=64):
    dataset = tf.contrib.data.TextLineDataset(filename)
    dataset = dataset.skip(1)
    dataset = dataset.shuffle(20000)
    feature_dataset = dataset.map(parse_fetures)
    label_dataset = dataset.map(parse_left_label)
    train_set = feature_dataset.take(14000).repeat(3).batch(minibatch_size)
    train_label_set = label_dataset.take(14000).repeat(3).batch(minibatch_size)
    test_set = feature_dataset.skip(14000).batch(1000)
    test_label_set = label_dataset.skip(14000).batch(1000)
    return train_set, train_label_set, test_set, test_label_set

def init_parameters(dims:list):
    w = {}
    b = {}
    layers = len(dims);
    for i in range(1,layers):
        w[i] = tf.multiply(tf.get_variable(name="W"+str(i), shape=[dims[i],dims[i-1]], dtype=tf.float32, initializer=tf.random_normal_initializer()),0.01)
        b[i] = tf.get_variable(name="b"+str(i), shape=[dims[i],1], dtype=tf.float32, initializer=tf.zeros_initializer())
    return w,b

def backward(W,b,X):
    a = X
    for i in range(1,len(W)):
        Z = tf.matmul(W[i],a)+b[i]
        tf.nn.dropout(Z,keep_prob=0.7)
        a = tf.nn.relu(Z)
    y_hat = tf.matmul(W[i+1],a)
    tf.summary.scalar("y_hat",tf.norm(y_hat))
    return y_hat

def compute_cost(y_hat,Y):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat,labels=Y)
    cost = tf.reduce_mean(loss)
    return cost


def compute_accuracy(y_hat, test_y):
    predict = tf.cast(tf.greater(y_hat, 0.5), tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predict, test_y), tf.float32))


tf.reset_default_graph()
train, train_label, test, test_label = loadDataset("hr_comma_sep.csv", minibatch_size=32)
next_train = train.make_one_shot_iterator()
next_train_label = train_label.make_one_shot_iterator()
next_test = test.make_one_shot_iterator()
next_test_label = test_label.make_one_shot_iterator()
W, b = init_parameters([7, 7, 6, 5, 5, 4, 1])
X = tf.placeholder(dtype=tf.float32, shape=[7, None])
Y = tf.placeholder(dtype=tf.float32, shape=[1, None])
y_hat = backward(W, b, X)
cost = compute_cost(y_hat, Y)
tf.summary.scalar("cost", cost)
train = tf.train.AdamOptimizer(0.01).minimize(cost)
accuracy = compute_accuracy(y_hat, Y)
tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # writer = tf.summary.FileWriter(graph=sess.graph, logdir="./")
    sess.run(tf.global_variables_initializer())
    i = 0
    train_cost_list = []
    train_accuracy_list = []
    while True:
        try:
            x_batch = sess.run(next_train.get_next()).T
            y_batch = sess.run(next_train_label.get_next()).T
            sess.run(train, feed_dict={X: x_batch, Y: y_batch})
            i = i + 1
            if i % 10 == 0:
                # writer.add_summary(sess.run(tf.summary.merge_all(), feed_dict={X: x_batch, Y: y_batch}), i)
                train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={X: x_batch, Y: y_batch})
                train_cost_list.append(train_cost)
                train_accuracy_list.append(train_accuracy)
                print("train cost:" + str(train_cost), "train accuaracy" + str(train_accuracy))
        except tf.errors.OutOfRangeError:
            break;
    test_batch = sess.run(next_test.get_next()).T
    test_label_batch = sess.run(next_test_label.get_next()).T
    test_cost, test_accuracy = sess.run([cost, accuracy], feed_dict={X: test_batch, Y: test_label_batch})
    plt.plot(train_cost_list, label="cost")
    plt.plot(train_accuracy_list, label="accuracy")

    print("test cost:" + str(test_cost), "test accuaracy" + str(test_accuracy))
    plt.show()
# print(sess.run(W))
