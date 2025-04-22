import numpy as np
import mediapipe as mp
import cv2
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import math as math
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volper = 0
volRange=volume.GetVolumeRange()
minVol , maxVol , volPer= volRange[0], volRange[-1], 0
class Handtracking:
    def __init__(self,mode=False, limit_hands=2,detection=0.5,tracktion=0.5):
        self.__mode__ = mode
        self.__maxHands__ = limit_hands
        self.__detectionCon__ = detection
        self.__trackCon__ = tracktion
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
    # to locate different fingers
    def locatefingers(self,frame,draw=True):
        convt_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(convt_image)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand, self.handsMp.HAND_CONNECTIONS)
        return frame
    # To track every finger movement by drawing tracking points and a bounding box
    def trackposition(self,frame,handNo=0,draw=True):
       x_cords=[]
       y_cords=[]
       bounding_box=[]
       self.lmslist=[]
       if self.results.multi_hand_landmarks:
           myhand=self.results.multi_hand_landmarks[handNo]
           for id,lm in enumerate(myhand.landmark):
               h , w , c=frame.shape
               cx,cy = int(lm.x*w),int(lm.y*h)
               x_cords.append(cx)
               y_cords.append(cy)
               self.lmslist.append([id,cx,cy])
               if draw: cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
           xmin,xmax= min(x_cords),max(x_cords)
           ymin,ymax= min(y_cords),max(y_cords)
           bounding_box = xmin,ymin,xmax,ymax
           print("Hands Keypoint")
           print(bounding_box)
           print(lmslist)
           if draw:
               cv2.rectangle(frame,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,255,0),2)
       return self.lmslist,bounding_box

    def findFingerUp(self):
        fingers = []

        if self.lmslist[self.tipIds[0]][1] > self.lmslist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmslist[self.tipIds[id]][2] < self.lmslist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
    def Volumecontol(self,frame):
        if len(self.lmslist) != 0:
            x1, y1 = self.lmslist[4][1], self.lmslist[4][2]  # thumb
            x2, y2 = self.lmslist[8][1], self.lmslist[8][2]  # index finger
            # Marking Thumb and Index finger
            cv2.circle(frame, (x1, y1), 15, (255, 255, 255))
            cv2.circle(frame, (x2, y2), 15, (255, 255, 255))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 10:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            vol = np.interp(length, [10, 440], [minVol, maxVol])
            return volume.SetMasterVolumeLevel(vol, None)
            volPer = np.interp(length, [50, 220], [0, 100])
co=cv2.VideoCapture(0)
co.set(cv2.CAP_PROP_FRAME_WIDTH,640)
co.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector= Handtracking()
lmslist=[]
if not co.isOpened():
    print("Error accesing camera plz check for any access related issue or a camera shutter blocking ")
    exit()
while True:
    ret,frame = co.read()
    frame = detector.locatefingers(frame)
    lmslist= detector.trackposition(frame)
    lmslist=detector.Volumecontol(frame)
    print(lmslist)
    frame=cv2.flip(frame,1)
    cv2.imshow('image',frame)
    if cv2.waitKey(1) == ord('q'): break
