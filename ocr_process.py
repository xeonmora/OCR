from Text_Detection import test as text_detect
import os, shutil
from Text_Detection.sort_image_crop import Sort
from Text_Recognization import test as text_recognize

detect = text_detect.TextDetect()

if not os.path.isdir('./Text_Detection/crop_images'):
    os.mkdir('./Text_Detection/crop_images')
else:
    shutil.rmtree('./Text_Detection/crop_images')
    os.mkdir('./Text_Detection/crop_images')

if not os.path.isdir('./Text_Detection/result'):
    os.mkdir('./Text_Detection/result')
else:
    shutil.rmtree('./Text_Detection/result')
    os.mkdir('./Text_Detection/result')

if not os.path.isdir('./Text_Recognization/demo_images/'):
    os.mkdir('./Text_Recognization/demo_images/')
else:
    shutil.rmtree('./Text_Recognization/demo_images/')
    os.mkdir('./Text_Recognization/demo_images/')

detect.text_detect()

for i in os.listdir('Text_Detection/data'):
    sort = Sort()
    sort.rowAlign('./Text_Detection/result/res_' + str(i).split(".")[0] + '.txt')
    sort.finalAlign()
    sort.imageCrop('./Text_Detection/data/' + str(i), str(i).split(".")[0])
    del sort

for i in os.listdir('./Text_Detection/crop_images'):
    print("Image Name:- ", str(i) + ".jpg")

    for j in os.listdir('./Text_Detection/crop_images/' + str(i)):
        shutil.copy('./Text_Detection/crop_images/' + str(i) + "/" + j, './Text_Recognization/demo_images/')

    text_line = text_recognize.text_recognize()
    print(text_line)
    shutil.rmtree('./Text_Recognization/demo_images/')
    os.mkdir('./Text_Recognization/demo_images/')
