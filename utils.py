import cv2
import numpy as np

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectCountour(countors ): 
    rectCon = []
    for i in countors:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if len(approx) == 4: 
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    return rectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02*peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)] # [0, 0]
    myPointsNew[3] = myPoints[np.argmax(add)] # [w, h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # [w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)] # [0, h]
    return myPointsNew       

def splitBoxes(img):
    columns = np.hsplit(img, 4)  # 4 columns
    boxes = []

    for column in columns:
        rows = np.vsplit(column, 25)  # 25 rows in each column
        for row in rows:
            # Crop the row to exclude the question index area
            cropped_row = row[:, int(row.shape[1] * 0.20):]
            # Split the cropped row into 4 bubbles
            bubbles = np.hsplit(cropped_row, 4)
            # Append each bubble to the list of boxes
            boxes.extend(bubbles)

    return boxes

def process_boxes(boxes, threshold=200):
    marked_bubbles = []
    questions = []

    for i, box in enumerate(boxes):
        non_zero_pixels = cv2.countNonZero(box)
        if non_zero_pixels > threshold:
            marked_bubbles.append(i)
        
        if (i+1) % 4 == 0:
            question_index = (i // 4) + 1
            marked_options = []
            for j in range(i-3, i+1):
                if j in marked_bubbles:
                    marked_options.append(chr(65 + (j % 4)))
            skipped = len(marked_options) == 0
            questions.append({'index': question_index, 'marked': marked_options, 'skipped': skipped})
            marked_bubbles = []

    return questions
