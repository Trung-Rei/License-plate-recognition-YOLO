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
img = resize_with_ratio(img, 1000/img.shape[0])
for bbox, label in pred_results:
    color = (0,255,255)
    draw_bbox(img, label, get_real_bbox(bbox, img.shape[1], img.shape[0]), color, 2)
cv2.imwrite('result.jpg', img)