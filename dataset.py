import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import IMAGE_SIZE


def center_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def data_split(image_dir, ann_dir):
    images = glob.glob(image_dir + "*")

    annotation = glob.glob(ann_dir + "*")

    img_dict = {i.split("/")[-1].split(".")[0]: i for i in images}

    ann_dict = {i.split("/")[-1].split(".")[0].rstrip("_segmentation"): i for i in annotation}

    concat_dict = {}
    for key, value in img_dict.items():
        for k, v in ann_dict.items():
            if k == key:
                concat_dict[value] = v

    img_list = []
    ann_list = []
    for im, an in concat_dict.items():
        img_list.append(cv2.resize(cv2.imread(im), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))
        ann_list.append(cv2.resize(cv2.imread(an, 0), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))

    img_arr = np.array(img_list)
    ann_arr = np.array(ann_list)
    ann_arr = ann_arr / 255

    X_Train, X_Test, y_Train, y_Test = train_test_split(img_arr, ann_arr, test_size=0.2, random_state=42)

    X_Train = center_normalize(X_Train)
    X_Test = center_normalize(X_Test)

    return X_Train, X_Test, y_Train, y_Test












