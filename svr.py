from flask import Flask, render_template, send_file, request
import os
from LPRecogniser import LPRecogniser
import cv2
from utils import *

lp_reco = LPRecogniser()

def predict(img_path):
    pred_results = lp_reco.predict(img_path)

    img = cv2.imread(img_path)
    img = resize_with_ratio(img, 1000/max(img.shape[:2]))

    bbox_only = [bbox for bbox, _ in pred_results]
    real_bboxes = recover_bbox(img, bbox_only)
    label_only = [lab for _, lab in pred_results]

    for bbox, label in zip(real_bboxes, label_only):
        color = (0,255,255)
        draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 2)
    cv2.imwrite('static/result.jpg', img)

# Khoi tao flask
app = Flask(__name__, template_folder="templates", static_folder="static")

# cau hinh thu muc upload
app.config["UPLOAD_FOLDER"] = "static"

# xu ly request
@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # lay anh upload
        img_file = request.files["file"]
        path_to_save = os.path.join(app.config["UPLOAD_FOLDER"], "in_img.jpg")
        img_file.save(path_to_save)

        # dua qua model
        predict(path_to_save)
        
        # tra ve ket qua
        return send_file("static/result.jpg", mimetype="image/gif")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)