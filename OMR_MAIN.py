import cv2 as cv
import numpy as np 
import utlis

width = 700
height = 700

# preprocessing
img = cv.imread('1.jpg')
img  = cv.resize(img,(width,height))
imgContours = img.copy()
imgBiggestContours = img.copy()

# Gray=0.299×Red+0.587×Green+0.114×Blue
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),1)
canny = cv.Canny(blur,10,50)
blank = np.zeros(gray.shape,dtype=np.uint8)

# Finding all Contours 
contours, heirarchies = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgContours,contours,-1,(0,255,0),2)

# Fiding rectangle
rectContours = utlis.rectangleContours(contours)
biggestRect = utlis.getCornerPoints(rectContours[0])
gradeBox = utlis.getCornerPoints(rectContours[1])

if biggestRect.size != 0 and  gradeBox.size != 0:
    cv.drawContours(imgBiggestContours,biggestRect,-1,(0,255,0),20)
    cv.drawContours(imgBiggestContours,gradeBox,-1,(255,0,0),20)
    
    biggestRect = utlis.reorder(biggestRect)
    gradeBox = utlis.reorder(gradeBox)

    pt1 = np.float32(biggestRect)
    pt2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(pt1,pt2)
    imgBigRect = cv.warpPerspective(img,matrix,(width,height))

    ptG1 = np.float32(gradeBox)
    ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG = cv.getPerspectiveTransform(ptG1,ptG2)
    imgGradeDisplay = cv.warpPerspective(img,matrixG,(325,150))
    cv.imshow('Grade',imgGradeDisplay)

# Applying Threshold
imgBigRect_Gray = cv.cvtColor(imgBigRect,cv.COLOR_BGR2GRAY)
ret , imgThresh = cv.threshold(imgBigRect_Gray,170,255,cv.THRESH_BINARY_INV)

# Spliting the box to get the grade circle
gradeCircle = utlis.splitBoxes(imgThresh)
# cv.imshow('test',gradeCircle[2])


# getting nonzero pixelvalues
pixelValues = np.zeros((5,5))
countCol = 0
countRow = 0

for img in gradeCircle:
    totalPixels = cv.countNonZero(img)
    pixelValues[countRow][countCol] = totalPixels
    countCol += 1
    if  countCol == 5:
        countCol = 0
        countRow += 1
# print(pixelValues)

maxIndex = []
for x in range(0,5):
    arr = pixelValues[x]
    index_max = np.where(arr==np.amax(arr))
    maxIndex.append(index_max[0][0])

# print(maxIndex)


# Displaying
# imageArray = ([img,gray,blur,canny],
#               [imgContours,imgBiggestContours,imgBigRect,imgThresh]) 
# imgStacked = utlis.stackImages(imageArray,0.5)

# cv.imshow('Stacked Images',imgStacked)

cv.waitKey(0)