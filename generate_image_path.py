import os
 
data_path = '/home/qcb/ImageHubTrain/Debris_YOLO/images/val/'
img_names = os.listdir(data_path)
 
list_file = open('/home/qcb/ProjectQin/yolov5-3.1/data/debris_val2017.txt', 'w')
index = 0
for img_name in img_names:
    list_file.write(data_path + '%s\n'%img_name) 
    print("current index: ", index)   
    index = index + 1
list_file.close()