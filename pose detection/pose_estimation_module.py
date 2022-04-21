import cv2 
import mediapipe as mp
import time 


class PoseDetection():

    def __init__(self,mode = False, upperBody = False, smooth = True, detectionConfidence = 1, trackingConfidence = 1):

        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def FindPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
        
#  for id, landmark in enumerate(results.pose_landmarks.landmark):
#             height, width, channel = img.shape
#             print(id,landmark)
#             cx, cy = int(landmark.x * width), int(landmark.y * height)
#             cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = PoseDetection()

    while True:
        success, img = cap.read()
        img = detector.FindPose(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()