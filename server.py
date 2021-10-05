from http.server import BaseHTTPRequestHandler
import os
import cv2
from utils import *
from LPRecogniser import LPRecogniser

path = 'web_server/test.jpg'

lp_reco = LPRecogniser()

def run():
    pred_results = lp_reco.predict(path)

    img = cv2.imread(path)
    img = fit_to_square(img, size=1000)
    for bbox, label in pred_results:
        color = (0,255,255)
        draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 2)
    cv2.imwrite('web_server/result.jpg', img)

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/web_server/uploader.php'
        try:
            split_path = os.path.splitext(self.path)
            request_extension = split_path[1]
            if request_extension != ".py":
                f = open(self.path[1:]).read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes(f, 'utf-8'))
            else:
                f = "File not found"
                self.send_error(404,f)
        except:
            f = "File not found"
            self.send_error(404,f)

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself

        post_data = post_data.split(b'\r\n\r\n')[1:]
        post_data = b''.join(post_data)
        with open('web_server/test.jpg', 'wb') as f:
            f.write(post_data)
        
        run()

        with open('web_server/result.jpg', 'rb') as f:
            re_data = f.read()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(re_data)

