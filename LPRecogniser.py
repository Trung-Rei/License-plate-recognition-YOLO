from CharClassifier import CharClassifier
from YOLODetector import YOLODetector
import cv2
import numpy as np
from utils import *

class LPRecogniser:
    def __init__(self) -> None:
        self.lp_det = YOLODetector('lp_detection/yolov4-tiny-lp-det_best.weights', 'lp_detection/yolov4-tiny-lp-det.cfg')
        self.chr_det = YOLODetector('character_detection/yolov4-tiny-char-detect_last.weights', 'character_detection/yolov4-tiny-char-detect.cfg', confi_thres=0.7)
        self.chr_cls = CharClassifier('character_recognition/myCNN_backup_28_BN_new.h5')

    def predict(self, im_path):
        img = cv2.imread(im_path)

        # Detect license plates
        img = fit_to_square(img, size=max(img.shape[:2]))
        lp_bboxes = self.lp_det.detect(img)
        if len(lp_bboxes) == 0:
            return []

        license_plates = [] # cropped LP with original size
        license_plates_crop_resize = [] # cropped LP in square
        for bbox in lp_bboxes:
            crop = crop_im(img, bbox)

            crop_resize = resize_with_ratio(crop, 100/crop.shape[1])
            blr = cv2.GaussianBlur(crop_resize, (5,5), 1.0)
            license_plates.append(blr)

            crop_resize = cv2.resize(crop, (100, 100))
            blr = cv2.GaussianBlur(crop_resize, (5,5), 1.0)
            license_plates_crop_resize.append(blr)
        
        # Detect and recognize characters
        result = []
        for lp_im, lp_im_ori in zip(license_plates_crop_resize, license_plates):
            chr_bboxes = self.chr_det.detect(lp_im)
            if len(chr_bboxes) == 0:
                result.append('')
                continue

            characters = []
            char_centers = []
            for bbox in chr_bboxes:
                # Padding bbox
                pad_bbox = bbox.copy()
                pad_bbox[2] += 0.05
                pad_bbox[3] += 0.05
                new_im = crop_im(lp_im_ori, pad_bbox)
                
                # Character should be portrait
                if new_im.shape[1] > new_im.shape[0]:
                    continue

                # fit character to square for recognising
                new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
                new_im = resize_with_ratio(new_im, 28/new_im.shape[0])
                border = 28-new_im.shape[1]
                if border % 2 == 0:
                    border_L = border_R = border // 2
                else:
                    border_L = border // 2
                    border_R = border_L + 1
                new_im = cv2.copyMakeBorder(new_im, 0, 0, border_L, border_R, cv2.BORDER_REPLICATE)
                
                characters.append(new_im)
                char_centers.append(bbox[:2])
            
            pred_chars = self.chr_cls.predict(characters)
            sorted_chars = self.format_LP(pred_chars, char_centers)
            result.append(''.join(sorted_chars))

        return [lp for lp in zip(lp_bboxes, result)]

    def format_LP(self, chars, char_centers):
        x = [c[0] for c in char_centers]
        y = [c[1] for c in char_centers]
        y_mean = np.mean(y)

        if y_mean - min(y) < 0.1:
            return [i for _, i in sorted(zip(x, chars))]
        
        sorted_chars = [i for _, i in sorted(zip(x, chars))]
        y = [i for _, i in sorted(zip(x, y))]
        first_line = [i for i in range(len(chars)) if y[i] < y_mean]
        second_line = [i for i in range(len(chars)) if y[i] > y_mean]
        return [sorted_chars[i] for i in first_line] + ['-'] + [sorted_chars[i] for i in second_line]
