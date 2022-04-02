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
    if not os.path.exists(json_path):
        print("\n ori json_path is not exists!!!!")
        return 0
    flag=np.zeros((1,10),dtype=int)
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
    print(image_save_path,label_save_path)

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
    jsonList = []
    m=0
    fout=open('error.txt','w')
    for root, dirs, files in os.walk(json_path, topdown=True):
        if root.__contains__('*'):
            continue
        max_steps = len(files)
        process_bar = ShowProcess(max_steps)
        for f in files:
            process_bar.show_process()
            if f.endswith('.json'):
                if (f in jsonList):
                    m += 1
                    continue
                file_path = os.path.join(root, f)

                image_path = file_path
                #print(files,image_path)
                if os.path.isfile(image_path):
                    with open(image_path, 'r') as load_f:
                        try:
                            json_file = json.load(load_f)

                        except:
                            continue
                json_file = json.load((open(file_path)))
                if False: #json_file['imageData'] is not None:
                    image = utils.img_b64_to_arr(json_file['imageData'])
                    print('aaffffffffffffffff')
                else:
                    #print('kk')
                    try:
                        imgPath=os.path.join(root, json_file['imagePath'])
                        if(not os.path.exists(imgPath)):
                            if(str(imgPath).find('.png')>0):
                                imgPath=str(imgPath).replace('.png','.PNG')
                            elif(str(imgPath).find('.bmp')>0):
                                imgPath = str(imgPath).replace('.bmp', '.BMP')
                            elif(str(imgPath).find('.PNG')>0):
                                imgPath = str(imgPath).replace('.PNG', '.png')
                            elif (str(imgPath).find('.BMP') > 0):
                                imgPath = str(imgPath).replace('.BMP', '.bmp')


                        image = Image.open(imgPath)
                        #
                    except:
                        print('error',imgPath)
                        fout.write(imgPath+'\n')
                        continue
                        #exit(-1)
                image = np.uint8(image)
                # 保存image
                im = Image.fromarray(np.uint8(image))

                image_save_path_save = os.path.join(image_save_path, f)
                image_save_path_save = image_save_path_save.replace(
                    ".json", ".png")
                if os.path.exists(image_save_path_save):
                    print('yy',image_save_path_save)
                    exit(-1)
                im.save(image_save_path_save)

                # 写入坐标
                path_txt = os.path.join(label_save_path, f)
                path_txt = path_txt.replace('.json', '.txt')
                #print('kk',path_txt)
                shapes = json_file['shapes']
                count = 0
                height = json_file['imageHeight']
                width = json_file['imageWidth']
                if(im.width!=width or im.height!=height):
                    print(im.width,im.height)
                    print('wh is erro',imgPath)
                    exit(-1)
                # shapes
                shape_save_path = os.path.join(image_save_path, 'images.shape')
                with open(shape_save_path, 'a') as file:
                    file.write('%g, %g\n' % (width, height))
                if os.path.exists(path_txt):
                    print(path_txt)
                    exit(-1)
                print('dd',image_save_path_save,path_txt)
                with open(path_txt, 'a') as ftxt:
                    for shape in shapes:
                        label = shape['label']
                        if label not in labels:
                            print(label,labels,path_txt)
                        points = shape['points']
                        shape_type = shape['shape_type']
                        if shape_type == "rectangle":
                            x_center, y_center, w, h = point_to_box(
                                points, height, width)
                        else:
                            x, y, w, h = cv2.boundingRect(
                                np.asarray(points).astype(np.float32))
                            x_center = (x + w / 2.0) / width
                            y_center = (y + h / 2.0) / height
                            w = 1.0*w / width
                            h = 1.0*h / height
                            

                        if label in labels:
                            ftxt.write('%g %.8f %.8f %.8f %.8f\n' % (int(labels[label]), x_center,y_center,w,h))
                            count += 1
                        else:
                            print('label: {} 已忽略'.format(label))
                jsonList.append(f)

                # 保存label
                try:
                    for shape in json_file['shapes']:
                        label_statistic[shape['label']
                                        ] = label_statistic[shape['label']] + 1
                        shape_num = shape_num + 1
                except:
                    continue
                json_num += 1
    process_bar.close()

    statistic_path = os.path.join(dataset_save_path, 'statistic.cvs')

    # 使用字典创建一个DataFrame
    dic = {

        'value': [],

    }
    df = DataFrame(data=dic)
    df.loc['json_num'] = [json_num]
    df.loc['shape_num'] = [shape_num]

    label_min = 10000000.0
    label_min_nonzero = 10000000.0
    for key in label_statistic:
        if label_statistic[key] < label_min and labels[key] > 0:
            label_min = label_statistic[key]
        if label_statistic[key] < label_min and label_statistic[key] > 0:
            label_min_nonzero = label_statistic[key]
    
    print(label_min)
    print(label_min_nonzero)
    df.loc['label_min'] = [label_min]
    df.loc['label_min_nonzero'] = [label_min_nonzero]
    for key in label_statistic:
        if int(shape_num) != 0:
            if label_statistic[key] > 0:
                df.loc[key] = label_statistic[key]
    df.to_csv(statistic_path, index=True, header=True)
    fout.close()


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
