import math
import shutil
import sys
import os
import argparse
import random

from utils.util import read_json, ShowProcess

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Segmentation prepare')
    args.add_argument('--trainImgDir', default='E:/Seafile/xietouTrData/images/train',
                      type=str,help='')
    args.add_argument('--trainLabelDir', default='E:/Seafile/xietouTrData/labels/train',
                      type=str, help='')
    args.add_argument( '--valImgDir', default='E:/Seafile/xietouTrData/images/val',
                       type=str, help='')
    args.add_argument('--valLabelDir', default='E:/Seafile/xietouTrData/labels/val',
                      type=str, help='')

    args.add_argument('--val_radio',
                      default='0.2',
                      type=float, help='')
    args = args.parse_args()
    trainImgDir = args.trainImgDir
    val_radio=args.val_radio
    train_pngs=[]
    for root, dirs, files in os.walk(trainImgDir, topdown=True):
        if root.__contains__('*'):
            continue
        max_steps = len(files)
        process_bar = ShowProcess(max_steps)
        for f in files:
            process_bar.show_process()
            if f.endswith('.png'):
                train_png_path = os.path.join(root, f)
                train_pngs.append(train_png_path)

        process_bar.close()
    file_num=len(train_pngs)
    val_list = random.sample(range(file_num), math.floor(val_radio * file_num))
    try:
        if not os.path.exists(args.valImgDir):
            os.makedirs(args.valImgDir)
        if not os.path.exists(args.valLabelDir):
            os.makedirs(args.valLabelDir)
        for i in val_list:
            shutil.move(train_pngs[i],args.valImgDir)
            shutil.move(str(train_pngs[i]).replace('images','labels').replace('.png','.txt'),args.valLabelDir)
            print(train_pngs[i])
    except Exception as e:
        print(e)




    print('ok')