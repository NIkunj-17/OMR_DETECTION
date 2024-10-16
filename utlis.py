import cv2 as cv
import numpy as np 

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, fx=scale, fy=scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])

        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)

        for d in range(0, rows):
            for c in range(0, cols):
                cv.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                             (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                             (255, 255, 255), cv.FILLED)
                cv.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver



def rectangleContours(contours):
    
    rectContour = []
    for i in contours:
        area = cv.contourArea(i)
        # print("Area",area)
        if area>50:
            peri = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,0.02*peri,True)
            # print(len(approx))
            if len(approx) == 4:
                rectContour.append(i)
        rectContour.sort(key=cv.contourArea,reverse=True)
    
    return  rectContour

def getCornerPoints(cont):
    peri = cv.arcLength(cont,True)
    approx = cv.approxPolyDP(cont,0.02*peri,True)
    return approx


def reorder(points):
    points = points.reshape((4,2))
    pointsNew = np.zeros((4,1,2),np.int32)
    # print(points)
    add= points.sum(1)
    # print(add)
    pointsNew[0] = points[np.argmin(add)]  # top-left point
    pointsNew[3] = points[np.argmax(add)]   # bottom-right point
    
    diff = np.diff(points,axis=1)
    pointsNew[1] = points[np.argmin(diff)]  
    pointsNew[2] = points[np.argmax(diff)]

    return pointsNew

def splitBoxes (img):
    rows = np.vsplit(img,5)
    gradeCircle = []
    for r in rows:
        cols = np.hsplit(r,5)
        for circle in cols:
            gradeCircle.append(circle) 
    
    return gradeCircle

