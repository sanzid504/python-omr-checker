from flask import jsonify, request, Flask
import requests
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import utils

app = Flask(__name__)

@app.route('/', methods=['POST'])
def detect():
    try:
        req_body = request.get_json()
        if 'file' not in req_body:
            return jsonify({'message': 'Failed', 'error': 'Missing file in request body'})
        
        print('image url: ', req_body['file'])
        response = requests.get(req_body['file'])
        if response.status_code != 200:
            return jsonify({'message': 'Failed', 'error': 'Failed to fetch image'})
        
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        ## pre-processing

        img = cv2.resize(img, (700,700))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



        ## decode qr-code 
        
        qr_code = decode(imgGray)
        if len(qr_code) > 0:
            print(qr_code[0].data.decode('utf-8'))
        
        ## other processing        
        
        imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)
        
        # Finding all contours

        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find rectangles

        rectCon = utils.rectCountour(contours)
        biggestContour = rectCon[0]
        biggestContour = utils.getCornerPoints(biggestContour)
        
        # Draw contours for display purposes only
        imgDisplayContours = img.copy()
        cv2.drawContours(imgDisplayContours, biggestContour, -1, (0, 255, 0), 10)

        if biggestContour.size != 0 :
            biggestContour = utils.reorder(biggestContour)
        
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0], [700,0], [0,700], [700,700]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (700,700))

            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThresh)
            ques = utils.process_boxes(boxes)
            # print(ques)
            return jsonify({'message': 'Success', 'data': ques, 'qr_code': qr_code[0].data.decode('utf-8')})
        else:
            return jsonify({'message': 'Failed', 'data': 'No contour detechted', 'qr_code': qr_code[0].data.decode('utf-8')})

    except Exception as e:
        return jsonify({'message': 'Failed', 'error': str(e)})

if __name__ == "__main__":
    app.run(port=4040)
