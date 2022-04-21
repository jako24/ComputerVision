import cv2
from cv2 import COLOR_BGR2RGB
import mediapipe as mp 
import time

cap = cv2.VideoCapture(0)

pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            boundingboxClass = detection.location_data.relative_bounding_box
            imageheight, imagewidth, imagechannel = img.shape
            bbox = int(boundingboxClass.xmin * imagewidth), int(boundingboxClass.ymin * imageheight), int(boundingboxClass.width * imagewidth), int(boundingboxClass.height * imageheight)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1] -20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)