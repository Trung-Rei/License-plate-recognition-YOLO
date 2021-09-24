from LPDetector import LPDetector
from CharDetector import CharDetector
from CharClassifier import CharClassifier
import cv2
import numpy as np

class LPRecogniser:
    def __init__(self) -> None:
        self.lp_det = LPDetector('weights_and_cfg_files/yolo-tinyv4-obj_last.weights', 'weights_and_cfg_files/yolo-tinyv4-obj.cfg')
        self.chr_det = CharDetector('weights_and_cfg_files/yolov4-tiny-char-detect_last.weights', 'weights_and_cfg_files/yolov4-tiny-char-detect.cfg')
        self.chr_cls = CharClassifier('weights_and_cfg_files/myCNN_backup_28_BN.h5')

    def predict(self, im_path):
        img = cv2.imread(im_path)

        lp_bboxes = self.lp_det.detect(img)
        if len(lp_bboxes) == 0:
            return []
        license_plates = []
        lp_bboxes_v2 = []
        for bbox in lp_bboxes:
            crop_resize = cv2.resize(self.crop_im(img, bbox), (100, 100))
            blr = cv2.GaussianBlur(crop_resize, (5,5), 1.0)
            license_plates.append(blr)
            lp_bboxes_v2.append(self.to_bbox_v2(bbox))

        result = []
        for lp_im in license_plates:
            chr_bboxes = self.chr_det.detect(lp_im)
            if len(chr_bboxes) == 0:
                result.append('')
                continue
            characters = []
            char_centers = []
            for bbox in chr_bboxes:
                pad_bbox = bbox.copy()
                pad_bbox[2] += 0.05
                pad_bbox[3] += 0.05
                new_im = self.crop_im(lp_im, pad_bbox)
                new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
                new_im = cv2.resize(new_im, (int(28*new_im.shape[1]/new_im.shape[0]*1.7), 28))
                border = 28-new_im.shape[1]
                if border % 2 == 0:
                    border_L = border_R = border // 2
                else:
                    border_L = border // 2
                    border_R = border_L + 1
                if border > 0:
                    new_im = cv2.copyMakeBorder(new_im, 0, 0, border_L, border_R, cv2.BORDER_REPLICATE)
                else:
                    new_im = cv2.resize(new_im, (28,28))
                characters.append(new_im)
                char_centers.append(bbox[:2])
            
            pred_chars = self.chr_cls.predict(characters)
            sorted_chars = self.format_LP(pred_chars, char_centers)
            result.append(''.join(sorted_chars))
        return [lp for lp in zip(lp_bboxes_v2, result)]

    def crop_im(self, img, bbox):
        ih, iw = img.shape[:2]
        cx, cy, w, h = tuple(bbox * np.array([iw, ih, iw, ih]))
        x1, y1 = int(cx-w/2), int(cy-h/2)
        x2, y2 = x1 + int(w), y1 + int(h)
        x1, y1 = max(0,x1), max(0,y1)
        new_img = img[y1:y2,x1:x2]
        return new_img

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

    def to_bbox_v2(self, bbox):
        cx, cy, w, h = tuple(bbox)
        x1, y1 = cx-w/2, cy-h/2
        x2, y2 = x1 + w, y1 + h
        return x1, y1, x2, y2
