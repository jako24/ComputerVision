import cv2
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
DRAWSPEC = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_TESSELATION, DRAWSPEC, DRAWSPEC)
            #468 points
            for id,lm in enumerate(faceLandmarks.landmark):
                # print(lm)
                imgHeight, imgWidth, imgChannel = img.shape
                x,y = int(lm.x* imgWidth), int(lm.y*imgHeight)
                print(id,x,y)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)