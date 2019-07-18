import model
import random
import tensorflow as tf
import argparse
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from config import IMAGE_SIZE
import dataset


def save_graph_to_file(sess, graph, graph_file_name):
    """
    Saves graph to a .pb model file.

    Args:
        sess: Current active TensorFlow Session.
        graph: Graph to be saved
        graph_file_name: Filename with which the graph has to be saved
    """
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["FINAL/conv2d/Conv2D"])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def main(_):
    v = vars(args)
    epochs = v['epochs']
    batch_size = v['batch_size']
    X_Train, X_Test, y_Train, y_Test = dataset.data_split(v['image_dir'],v['ann_dir'])
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE])
    pred = model.espnet(x, v['model_name'])

    # cross-entropy loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred, labels=y))

    train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    train_loss, val_loss = float("inf"), float("inf")
    temp_val_loss = val_loss
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()

    for i in range(epochs):
        total_train_loss, total_val_loss = 0, 0
        batches = int(len(X_Train) / batch_size)
        train_idx = [idx for idx in range(batches)]
        random.shuffle(train_idx)
        val_idx = [idx for idx in range(len(X_Test) / batch_size)]
        random.shuffle(val_idx)
        for j in train_idx:
            opt = sess.run([train_op], feed_dict={x: X_Train[j * batch_size:(j + 1) * batch_size, :],
                                                  y: y_Train[j * batch_size:(j + 1) * batch_size, :]})
            train_loss = sess.run([loss], feed_dict={x: X_Train[j * batch_size:(j + 1) * batch_size, :],
                                                     y: y_Train[j * batch_size:(j + 1) * batch_size, :]})
            total_train_loss = total_train_loss + train_loss[0]
            
        for k in val_idx:
            val_loss = sess.run([loss], feed_dict={x: X_Test[k * batch_size:(k + 1) * batch_size, :],
                                                   y: y_Test[k * batch_size:(k + 1) * batch_size, :]})
            total_val_loss = total_val_loss + val_loss[0]
        train_loss = (total_train_loss * 1.0) / batches
        val_loss = (total_val_loss * 1.0) / (len(X_Test) / batch_size)
        print "epoch:", (i + 1), "Train_Loss:", train_loss, "Validation_Loss:", val_loss
        if val_loss < temp_val_loss:
            temp_val_loss = val_loss
            save_graph_to_file(sess, sess.graph, v['model_file_name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, help='batch size', type=int)
    parser.add_argument('--model_name', default="espnet_c", help='Model name')
    parser.add_argument('--epochs', default=1, help='Number of epochs', type=int)
    parser.add_argument('--model_file_name', default="esp_net.pb", help='Name of the model filename')
    parser.add_argument('--image_dir', default='ISIC-2017_Training_Data/', help='Folder containing training images',
                        type=str)
    parser.add_argument('--ann_dir', default='ISIC-2017_Training_Part1_GroundTruth/',
                        help='Folder containing annotations of training images', type=str)

    args = parser.parse_args()
    tf.app.run()



