import cv2

def resize_with_ratio(img, ratio):
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
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
    return int(x1), int(y1), int(x2), int(y2)