import cv2
import numpy as np

class LPDetector:
    def __init__(self, weights_path, cfg_path):
        self.LP_model = cv2.dnn.readNet(weights_path, cfg_path)
        self.CONFI_THRES = 0.5

    def detect(self, in_img):
        height, width = in_img.shape[:2]

        blob = cv2.dnn.blobFromImage(in_img, 1/255.0, (416,416), 0, swapRB=True, crop=False)
        self.LP_model.setInput(blob)
        output_layers_name = self.LP_model.getUnconnectedOutLayersNames()
        layeroutput = self.LP_model.forward(output_layers_name)

        confidences = []
        boxes = []
        class_ids = []
        bboxes_yolo = []

        for output in layeroutput:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                if confidence > self.CONFI_THRES:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    detected_width = int(detection[2] * width)
                    detected_height = int(detection[3] * height)

                    x_min = int(center_x - detected_width / 2)
                    y_min = int(center_y - detected_height / 2)

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    bboxes_yolo.append(detection[:4])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.CONFI_THRES, nms_threshold=0.4)

        if len(indexes) > 0:
            bboxes_yolo = [bboxes_yolo[i] for i in indexes.flatten()]
        else:
            bboxes_yolo = []

        return bboxes_yolo