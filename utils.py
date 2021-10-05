import cv2
import numpy as np

def resize_with_ratio(img, ratio):
    w = round(img.shape[1] * ratio)
    h = round(img.shape[0] * ratio)
    return cv2.resize(img, (w, h))

def puttext_with_bg(img, text, pos, scale, thickness, t_color, bg_color, pad=0.8):
    font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    w, h = text_size
    padding = int(h * pad)
    cv2.rectangle(img, pos, (x + w + 2 * padding, y - h - 2 * padding), bg_color, -1)
    cv2.putText(img, text, (x + padding, y - padding), font, scale, t_color, thickness)

def draw_bbox(img, label, coord, color, scale):
    cv2.rectangle(img, coord[:2], coord[2:], color, scale)
    puttext_with_bg(img, label, coord[:2], scale, scale, t_color=(0,0,0), bg_color=color)

def get_real_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    return round(x1), round(y1), round(x2), round(y2)

def fit_to_square(img, size=416, bg_color=0, bboxes=None):
    if len(img.shape) > 2:
        bg_im = np.zeros([size,size,3],dtype=np.uint8)
    else:
        bg_im = np.zeros([size,size],dtype=np.uint8)

    bg_im[:] = bg_color
    
    ratio = size / max(img.shape[:2])
    w, h = round(img.shape[1]*ratio), round(img.shape[0]*ratio)
    fg_im = cv2.resize(img, (w, h))
    
    top_bord = int((size - h) / 2)
    left_bord = int((size - w) / 2)
    
    if len(img.shape) > 2:
        for c in range(3):
            bg_im[top_bord:top_bord+h, left_bord:left_bord+w, c] = fg_im[:, :, c]
    else:
        bg_im[top_bord:top_bord+h, left_bord:left_bord+w] = fg_im[:, :]

    if bboxes != None:
        new_bboxes = []
        for bb in bboxes:
            bbox = [round(i * ratio) for i in bb]
            new_bboxes.append([bbox[0]+left_bord, bbox[1]+top_bord,
                bbox[2]+left_bord, bbox[3]+top_bord])
        return bg_im, new_bboxes

    return bg_im

def yolo_to_bbox(img, yolo_fm):
    height, width = img.shape[:2]
    x, y, w, h = yolo_fm
    x1, x2 = x - w/2, x + w/2
    y1, y2 = y - h/2, y + h/2
    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    return [x1, y1, x2, y2]

def crop_im(img, bbox_yolo):
    ih, iw = img.shape[:2]
    cx, cy, w, h = tuple(bbox_yolo * np.array([iw, ih, iw, ih]))
    x1, y1 = round(cx-w/2), round(cy-h/2)
    x2, y2 = x1 + round(w), y1 + round(h)
    x1, y1 = max(0,x1), max(0,y1)
    new_img = img[y1:y2,x1:x2]
    return new_img

def read_yolo_content(path):
    boxes = []
    labels = []
    with open(path) as f:
        for line in f.readlines():
            box = [float(num) for num in line.split(' ')[:5]]
            boxes.append(box[1:])
            labels.append(box[0])
    
    return boxes, labels

def recover_bbox(ori_im, yolo_bboxes):
    square_size = max(ori_im.shape[:2])
    h, w = ori_im.shape[:2]

    top_bord = round((square_size - ori_im.shape[0]) / 2)
    left_bord = round((square_size - ori_im.shape[1]) / 2)

    ori_bboxes = []
    for bbox in yolo_bboxes:
        real_bbox = np.array(bbox) * square_size
        real_bbox[:2] = real_bbox[:2] - np.array([left_bord, top_bord])
        real_bbox = real_bbox / np.array([w, h, w, h])
        ori_bboxes.append(list(real_bbox))

    return ori_bboxes
