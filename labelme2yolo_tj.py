from pandas import Series, DataFrame
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import json
import cv2
import glob
from labelme import utils
import os
import shutil
import sys
import copy
from random import shuffle
import numpy as np
import time
from collections import OrderedDict
import argparse
from pathlib import Path
from utils.util import read_json, ShowProcess
from labelme import utils as lb_utils
# %%
import pandas as pd


def point_to_box(points, height, width):
    min_x = min_y = np.inf
    max_x = max_y = 0
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    w = max_x - min_x
    h = max_y - min_y
    c_x = min_x + w / 2.
    c_y = min_y + h / 2.
    # 归一化
    c_x_norm = c_x / width
    c_y_norm = c_y / height
    w_norm = w / width
    h_norm = h / height

    return (c_x_norm, c_y_norm, w_norm, h_norm)


def labelme2yolo(json_path, dataset_save_path, labels, type='train'):
    print(json_path)
    if not os.path.exists(json_path):
        print("\n ori json_path is not exists!!!!")
        return 0
    flag = np.zeros((1, 20), dtype=int)
    print(json_path, dataset_save_path)
    if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)
    image_save_path = os.path.join(dataset_save_path, 'images')
    label_save_path = os.path.join(dataset_save_path, 'labels')
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    if not os.path.exists(label_save_path):
        os.mkdir(label_save_path)
    image_save_path = os.path.join(dataset_save_path, 'images', type)
    label_save_path = os.path.join(dataset_save_path, 'labels', type)
    print(image_save_path, label_save_path)

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    # 生成list
    json_num = 0
    shape_num = 0

    label_statistic = copy.deepcopy(labels)
    for k, v in label_statistic.items():
        label_statistic[k] = 0
    n = 0
    m = 0
    jsonList = []
    for root, dirs, files in os.walk(json_path, topdown=True):
        if root.__contains__('*'):
            continue
        max_steps = len(files)
        process_bar = ShowProcess(max_steps)
        for f in files:
            process_bar.show_process()
            if f.endswith('.json'):
                n += 1
                if (f in jsonList):
                    m += 1
                    continue
                file_path = os.path.join(root, f)
                image_path = file_path
                print(n, file_path, image_path)
                if os.path.isfile(image_path):
                    with open(image_path, 'r') as load_f:
                        try:
                            json_file = json.load(load_f)
                            json_num += 1
                        except:
                            continue
                json_file = json.load((open(file_path)))

                shapes = json_file['shapes']
                for shape in shapes:
                    label = shape['label']
                    if label not in labels:
                        print(label, labels, file_path)
                        exit(-1)
                    points = shape['points']
                    shape_type = shape['shape_type']
                    height = json_file['imageHeight']
                    width = json_file['imageWidth']
                    if shape_type == "rectangle":
                        x_center, y_center, w, h = point_to_box(
                            points, height, width)
                    else:
                        x, y, w, h = cv2.boundingRect(
                            np.asarray(points).astype(np.float32))
                        x_center = (x + w / 2.0) / width
                        y_center = (y + h / 2.0) / height
                        w = 1.0 * w
                        h = 1.0 * h / height
                    # if(int(label)==8):
                    #     print(file_path)
                    #     exit(-1)
                    if label in labels:
                        # print(label)
                        # print(flag[0,1])
                        flag[0, int(label)] = flag[0, int(label)] + 1
                    else:
                        print('label: {} 已忽略'.format(label))
            jsonList.append(f)

    print(flag)
    print(n, m)


def main(args):
    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.config is None:
        print("\n config is not exists!!!!")
        return 0
    cfg_fname = Path(args.config)
    config = read_json(cfg_fname)

    json_path = config['labelme2yolo']['args']['json_path']
    dataset_save_path = config['labelme2yolo']['args']['dataset_save_path']
    labels = config['labelme2yolo']['args']['labels']
    type = config['labelme2yolo']['args']['type']

    print(json.dumps(config['labelme2yolo'],
                     sort_keys=True, indent=4, separators=(', ', ': ')))
    labelme2yolo(json_path, dataset_save_path, labels, type)


if __name__ == '__main__':
    print("start==========================")
    args = argparse.ArgumentParser(description='Segmentation prepare')
    args.add_argument('-c', '--config', default='F:\YoloWork\yolov5-6.1_LP\datapre\config_new.json', type=str,
                      help='config file path (default: None)')
    main(args)

    print("end===========================")

"""
xietou
F:\YoloWork\yolov5-6.1_LP\datapre\config_new.json
[[   0  852 1089   33   25    0   53    0    3    0    0    0    0    0
     0    0    0    0    0    0]]
1047 0
end===========================

"""