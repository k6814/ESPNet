import tensorflow as tf
import cv2
import numpy as np
import glob
import argparse
import scipy.misc
from config import IMAGE_SIZE


def center_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def main(_):
    v = vars(args)
    batch_size = v['batch_size']
    with tf.gfile.FastGFile(v['model_file_name'], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    test_list = glob.glob(v['input_folder'] + '*')

    test_img_list = []
    img_name = []
    for im in test_list:
        test_img_list.append(cv2.resize(cv2.imread(im), (IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_NEAREST))
        img_name.append(im)
    img_arr = center_normalize(np.array(test_img_list))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batches = int(len(img_arr) / batch_size)
        train_idx = [idx for idx in range(batches)]
        softmax_tensor = sess.graph.get_tensor_by_name('FINAL/conv2d/Conv2D:0')
        for j in train_idx:
            predictions = sess.run(softmax_tensor, {'Placeholder:0': img_arr[j * batch_size:(j + 1) * batch_size, :]})
            temp_name = img_name[j * batch_size:(j + 1) * batch_size]
            for (im_arr, temp) in zip(predictions, temp_name):
                img_prep = []
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        img_prep.append(np.argmax(softmax([im_arr[:, :, 0][i][j], im_arr[:, :, 1][i][j]])))
                scipy.misc.imsave(v['op_folder'] + temp.split('/')[-1], np.reshape(np.array(img_prep),
                                                                                   (IMAGE_SIZE, IMAGE_SIZE)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, help='batch size',
                        type=int)
    parser.add_argument('--input_folder', default='images/*/sample/',
                        help='Folder containing images to be tested')
    parser.add_argument('--op_folder', default='output/',
                        help='Output Folder')
    parser.add_argument('--model_file_name', default='esp_net.pb',
                        help='Name of the model filename')

    args = parser.parse_args()
    tf.app.run()
