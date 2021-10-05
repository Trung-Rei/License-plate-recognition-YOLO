from LPRecogniser import LPRecogniser
import cv2
from utils import *
import argparse
import time
from pathlib import Path

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./testIm.jpeg')
    return arg.parse_args()

args = get_arguments()
path = args.image_path

lp_reco = LPRecogniser()
start = time.time()
pred_results = lp_reco.predict(path)
stop = time.time()
print(round(stop-start, 2))

img = cv2.imread(path)
img = resize_with_ratio(img, 1000/max(img.shape[:2]))

bbox_only = [bbox for bbox, _ in pred_results]
real_bboxes = recover_bbox(img, bbox_only)
label_only = [lab for _, lab in pred_results]

for bbox, label in zip(real_bboxes, label_only):
    color = (0,255,255)
    draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 2)
cv2.imwrite('result.jpg', img)