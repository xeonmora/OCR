import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import Text_Detection.craft_utils as craft_utils
import Text_Detection.file_utils as file_utils
import Text_Detection.imgproc as imgproc
from Text_Detection.craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


result_folder = './Text_Detection/result/'

# Hyper parameters
canvas_size = 1280
mag_ratio = 1.5
show_time = False
test_folder = "./Text_Detection/data"
trained_model = "./Text_Detection/weights/craft_mlt_25k.pth"
link_threshold = 0.4
text_threshold = 0.7
low_text = 0.4
poly = False
cuda = True


def test_net(net, image, text_threshold, link_threshold, low_text, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)

    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


class TextDetect:
    def __init__(self):
        self.net = CRAFT()

        if cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

        if cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

    def text_detect(self):
        """ For test images in a folder """
        image_list, _, _ = file_utils.get_files(test_folder)
        print(image_list)

        t = time.time()

        # load data
        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text, poly)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

        print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    td = TextDetect()
    td.text_detect()
