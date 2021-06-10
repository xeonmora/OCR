import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class BigLetterDetection:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly images longer
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

        # load weights
        cfg.MODEL.WEIGHTS =  "./CharacterDetection/output/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model

        self.predictor = DefaultPredictor(cfg)

    def detect(self, image):
        img = cv2.imread(image)
        outputs = self.predictor(img)

        for i in list(list(outputs['instances'].__dict__.values())[1]['pred_boxes']):
            print(i)
            if 10 <= i[0] <= 130:
                print("big letter ", i)
                crop_image = img[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                cv2.imwrite('../OCR/temp.jpg', crop_image)
                return True
        else:

            return False

