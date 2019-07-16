import glob
import cv2
import numpy as np
import tensorflow as tf
import argparse
from config import IMAGE_SIZE


def mean_iou(ground_truth, prediction, num_classes):
    """
        Calculates mean Intersection-Over-Union (mIOU)

        Returns:
            iou: A Tensor representing the mean intersection-over-union.
            iou_op: An operation that increments the confusion matrix.
        """
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op


def main(_):
    v = vars(args)
    predicted_mask = glob.glob(v["prediction"] + "*")
    ground_truth_mask = glob.glob(v["ground_truth"] + "*")
    no_of_classes=v["number_of_classes"]
    predicted_mask_dict = {i.split("/")[-1].split(".")[0]: i for i in predicted_mask}
    ground_truth_mask_dict = {i.split("/")[-1].split(".")[0]: i for i in ground_truth_mask}

    predicted_gt_dict = {}
    for key, value in predicted_mask_dict.items():
        for k, v in ground_truth_mask_dict.items():
            if k == key:
                predicted_gt_dict[value] = v

    predicted_list = []
    gt_list = []
    for im, an in predicted_gt_dict.items():
        predicted_list.append(cv2.resize(cv2.imread(im, 0), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))
        gt_list.append(cv2.resize(cv2.imread(an, 0), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))

    predicted_arr = np.array(predicted_list)
    gt_arr = np.array(gt_list)
    gt_arr = gt_arr / 255
    predicted_arr = predicted_arr / 255
    iou, iou_op = mean_iou(tf.constant(predicted_arr, dtype=tf.int32), tf.constant(gt_arr, dtype=tf.int32),
                           no_of_classes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # need to initialize local variables for to use "tf.metrics.mean_iou"
        sess.run(
            tf.local_variables_initializer())

        sess.run(iou_op)
        print(iou.eval())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', default="ISIC-2017_Validation_Part1_GroundTruth/",
                        help='ground truth directory', type=str)
    parser.add_argument('--prediction', default="ISIC-2017_Validation_Data/",
                        help="Prediction Image Directory", type=str)
    parser.add_argument('--number_of_classes', default=2, help='No. of classes',type=int)
    args = parser.parse_args()
    tf.app.run()















