import warnings
warnings.filterwarnings(action="ignore")

import cv2
import numpy as np

class YOLODetector:
    def __init__(self, weights_path, cfg_path, confi_thres=0.5):
        self.model = cv2.dnn.readNet(weights_path, cfg_path)
        self.CONFI_THRES = confi_thres

    def detect(self, lp_img):
        height, width = lp_img.shape[:2]

        blob = cv2.dnn.blobFromImage(lp_img, 1/255.0, (416,416), 0, swapRB=True, crop=False)
        self.model.setInput(blob)
        output_layers_name = self.model.getUnconnectedOutLayersNames()
        layeroutput = self.model.forward(output_layers_name)

        confidences = []
        boxes = []
        bboxes_yolo = []

        for output in layeroutput:
            for detection in output:
                confidence = float(detection[5])
                if confidence > self.CONFI_THRES:
                    center_x = round(detection[0] * width)
                    center_y = round(detection[1] * height)
                    detected_width = round(detection[2] * width)
                    detected_height = round(detection[3] * height)

                    x_min = round(center_x - detected_width / 2)
                    y_min = round(center_y - detected_height / 2)

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    bboxes_yolo.append(detection[:4])
                    confidences.append(confidence)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.CONFI_THRES, nms_threshold=0.4)

        if len(indexes) > 0:
            bboxes_yolo = [bboxes_yolo[i] for i in indexes.flatten()]
        else:
            bboxes_yolo = []

        return bboxes_yolo