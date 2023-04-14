import cv2
import numpy as np
import utils
from pyzbar.pyzbar import decode

path = 'test_omr_3.png'
img = cv2.imread(path)

## pre-processing

img = cv2.resize(img, (700,700))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



## decode qr-code 
qr_code = decode(imgGray)
if len(qr_code) > 0:
    print(qr_code[0].data.decode('utf-8'))
    
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)
# Finding all contours

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find rectangles

rectCon = utils.rectCountour(contours)
biggestContour = rectCon[0]
biggestContour = utils.getCornerPoints(biggestContour)
gradePoints = utils.getCornerPoints(rectCon[1])

# Draw contours for display purposes only
imgDisplayContours = img.copy()
cv2.drawContours(imgDisplayContours, biggestContour, -1, (0, 255, 0), 10)
cv2.drawContours(imgDisplayContours, gradePoints, -1, (255, 0, 0), 10)

if biggestContour.size != 0 :
    biggestContour = utils.reorder(biggestContour)
    grade = utils.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0], [700,0], [0,700], [700,700]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (700,700))

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh)
    ques = utils.process_boxes(boxes)
    # print(ques)

