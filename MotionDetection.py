import cv2
import time
import numpy as np
import datetime
import os

# settings
WINDOW_NAME = 'MotionDetection'
ADDRESS='http://192.168.178.4/videostream.cgi?rate=0&user=admin&pwd=123456'
DIR_NAME = 'screenshots'

WIDTH = 640
HEIGHT = 480

DEFAULT_THRESHOLD = 25
DEFAULT_TRIGGER = 0.001

INTERVAL_CONNECT = 10
INTERVAL_HOLD = 5
INTERVAL_SHOT = 1

def printText(image, x, y, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,255,255)
    thickness = 1
    cv2.putText(image, text, (x, y), font, fontScale, color, thickness)

def printScreenshot(image):
    if(image is not None):
        dirname = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = datetime.datetime.now().strftime('%H-%M-%S')
        if(not os.path.isdir('./'+ DIR_NAME)):
            os.mkdir('./' + DIR_NAME)
        if(not os.path.isdir('./' + DIR_NAME + '/' + dirname)):
            os.mkdir('./' + DIR_NAME + '/' + dirname)
        index = 0
        fullPath = './' + DIR_NAME + '/' + dirname + '/' + filename + '.png'
        while(os.path.isfile(fullPath)):
            index += 1
            fullPath = './' + DIR_NAME + '/' + dirname + '/' + filename + '_' + str(index) + '.png'
        cv2.imwrite(fullPath, image)

# init
cv2.namedWindow(WINDOW_NAME)

videoCapture = cv2.VideoCapture()
inputImage = None
lastImage = None
diffImage = None

threshold = DEFAULT_THRESHOLD
trigger = DEFAULT_TRIGGER

connectionTries = 0
percentage = 0
detections = 0
screenshots = 0

previewThreshold = False
active = True
connected = False

timerConnect = time.time() - INTERVAL_CONNECT
timerHold = time.time() - INTERVAL_HOLD
timerShot = time.time()

# main loop
while(True):            
    if(not connected):
        if(time.time() - timerConnect >= INTERVAL_CONNECT):
            timerConnect = time.time()
            connectionTries += 1
            videoCapture.open(ADDRESS)
            if(videoCapture.isOpened()):
                connected = True
            else:
                videoCapture.release()
    elif(not videoCapture.isOpened()):
        connected = False
        videoCapture.release()
    else:
        returnValue, inputImage = videoCapture.read()
        if(returnValue == 0): # no image could be read
            connected = False
            videoCapture.release()
            
            inputImage = None
            lastImage = None
            diffImage = None
        else:
            connectionTries = 0
            grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
            if(lastImage is not None):
                diffImage = cv2.absdiff(grayImage,lastImage)
                _, diffImage = cv2.threshold(diffImage, threshold, 255, cv2.THRESH_BINARY)
                percentage = np.mean(diffImage.flatten()) / 255
                if(active and percentage >= trigger):
                    detections += 1
                    timerHold = time.time()
            lastImage = grayImage
            
            if(time.time() - timerHold <= INTERVAL_HOLD):
                if(time.time() - timerShot >= INTERVAL_SHOT):
                    timerShot = time.time()
                    screenshots += 1
                    printScreenshot(inputImage)
        
        previewImage = None
        if(not connected):
            previewImage = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            printText(previewImage, 5, 20, 'Connecting...')
            printText(previewImage, 5, 35, 'Tries: {}'.format(connectionTries))
        else:
            if(previewThreshold):
                if(diffImage is None):
                    previewImage = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                else:
                    previewImage = diffImage.copy()
            else:
                previewImage = inputImage.copy()            
        
            printText(previewImage, 5, 20, 'Threshold: {}'.format(threshold))
            printText(previewImage, 5, 35, 'Percent: {:.4f}'.format(percentage))
            printText(previewImage, 5, 50, 'Trigger: {:.4f}'.format(trigger))
            printText(previewImage, 5, 65, 'Detections: {}'.format(detections))
            printText(previewImage, 5, 80, 'Screenshots: {}'.format(screenshots))
            printText(previewImage, 5, 95, 'Active: {}'.format(active))
            
        cv2.imshow(WINDOW_NAME, previewImage)
            
    # read and handle pressed keys
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break
    elif(key == ord('+')):
        if(threshold < 255):
            threshold += 1
    elif(key == ord('-')):
        if(threshold > 1):
            threshold -= 1
    elif(key == ord('i')):
        if(trigger < 1.0):
            trigger += 0.0001
    elif(key == ord('d')):
        if(trigger > 0):
            trigger -= 0.0001
    elif(key == ord('s')):
        printScreenshot(inputImage)
    elif(key == ord('p')):
        previewThreshold = not previewThreshold
    elif(key == ord('a')):
        active = not active
        
videoCapture.release()
cv2.destroyAllWindows()