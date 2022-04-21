import cv2
import mediapipe as mp 
import time

#---------THIS IS WHAT WE ALWAYS DO FOR RUNNING A WEBCAM --------------------
# vedio object
cap = cv2.VideoCapture(0)   
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 
cTime = 0

while True:
    success, img = cap.read()     #our frame
    imgRBG = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlendmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handlendmarks.landmark):
                # print(id,landmark)
                #where it is in pixels
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                print(id,cx,cy)
                if id == 8:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlendmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()    #current time
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1) 

#-----------------------------------------------------------------------------
