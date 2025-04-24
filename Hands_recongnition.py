import autopy
import numpy as np
import mediapipe as mp
import cv2
import math as math
from ctypes import cast, POINTER
import time
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import io
from PIL import ImageGrab
import win32clipboard


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

class Handtracking:
    def __init__(self, mode=False, limit_hands=2, detection=0.7, tracktion=0.9):
        self.__mode__ = mode
        self.__maxHands__ = limit_hands
        self.__detectionCon__ = detection
        self.__trackCon__ = tracktion
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(static_image_mode=mode,
            max_num_hands=limit_hands,
            min_detection_confidence=detection,
            min_tracking_confidence=tracktion)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        #self.on_or_off = 0
        self.is_drawing = False
        self.draw_points =[]
        self.last_drawn_point = None
        self.current_points=[]
        self.last_screen_shot =  float(0)



# to locate different fingers and draw connections
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
               cz=lm.z
               x_cords.append(cx)
               y_cords.append(cy)
               self.lmslist.append([id,cx,cy,cz])
               if draw: cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
           xmin,xmax= min(x_cords),max(x_cords)
           ymin,ymax= min(y_cords),max(y_cords)
           bounding_box = xmin,ymin,xmax,ymax
           '''if draw:
               cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)'''
       return self.lmslist

    def findFingerUp(self):
        if not self.lmslist: return []
        fingers = []
        # thumb logic
        #  lmslist = [landmark_id,x,y]
        #  self.tipIds[0] is 4, which is the thumb tip landmark index.
        #  self.tipIds[0] - 1 is 3, which is the thumb's joint just before the tip
        #             x co-ord for thumb tip >  x co-ord for thumb mid
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


    def Volumecontrol(self,frame):
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
            volume.SetMasterVolumeLevel(vol, None)

    def draw_mode(self, frame, canvas):
        if not self.lmslist:
            self.is_drawing = False
            return canvas

        fingers = self.findFingerUp()
        h, w, _ = frame.shape

        # Toggle drawing mode on thumbs up
        if fingers ==  [0, 1, 0, 1, 1]:
            if self.is_drawing is None:
                print("Entered drawing mode.")
                self.is_drawing = False  # Ready to start drawing
            elif self.is_drawing:  # Exit drawing mode
                print("Exited drawing mode.")
                self.is_drawing = None
            return canvas

        # Inside drawing mode
        if self.is_drawing is not None:
            # Start/Continue drawing with 3 fingers (middle, ring, pinky)
            if fingers == [0, 0, 1, 1, 1]:
                if not self.is_drawing:
                    self.current_points = []
                    self.last_drawn_point = None
                    self.is_drawing = True

                x, y = self.lmslist[8][1], self.lmslist[8][2]
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))

                if self.last_drawn_point is None or math.hypot(x - self.last_drawn_point[0],
                                                               y - self.last_drawn_point[1]) > 2:
                    self.current_points.append((x, y))
                    self.last_drawn_point = (x, y)
            else:
                # Stop drawing and save the stroke
                if self.is_drawing and self.current_points:
                    self.draw_points.append(self.current_points)
                    self.current_points = []
                self.is_drawing = False

            # Draw previous strokes
            for points in self.draw_points:
                for i in range(1, len(points)):
                    cv2.line(canvas, points[i - 1], points[i], (0, 0, 255), 10)

            # Draw current stroke
            for i in range(1, len(self.current_points)):
                cv2.line(canvas, self.current_points[i - 1], self.current_points[i], (0, 0, 255), 10)

            # Clear canvas with thumb and pinky
            if fingers == [1, 0, 0, 0, 1]:
                print("Clearing canvas.")
                self.draw_points = []
                self.current_points = []
                self.last_drawn_point = None
                self.is_drawing = False
                canvas.fill(0)

        return canvas

    def screen_shot(self, frame):
        if not self.lmslist:
            return
        fingers = self.findFingerUp()
        current_time =time.time()
        if fingers == [0, 0, 0, 0, 0] :
            if int(abs(current_time - self.last_screen_shot)) > 2: #cooldown 2 sec
                img = ImageGrab.grab() #grabs the whole screens screenshot
                self.last_screen_shot = current_time

                #copy to clipboard
                output = io.BytesIO() #opens a memory byte stream
                img.convert("RGB").save(output, "BMP") # converts to bitmap image as windows only accepts this not rgb
                data = output.getvalue()[14:]  # BMP file header starts from 14th byte aas the real data starts from there
                output.close() # closes the memory stream

                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)#CF_DIB device independent bitmap
                win32clipboard.CloseClipboard()
                print("Screenshot copied to clipboard!")



    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        l1=self.lmslist[p1][1:]
        l2= self.lmslist[p2][1:]
        x1, y1 = l1[0],l1[1]
        x2, y2 = l2[0],l2[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]



co=cv2.VideoCapture(1)
co.set(cv2.CAP_PROP_FRAME_WIDTH,680)
co.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = Handtracking()
canvas = np.zeros((1280, 720, 3), dtype=np.uint8)

if not co.isOpened():
    print("Error accesing camera plz check for any access related issue or a camera shutter blocking ")
    exit()

width = 640             # Width of Camera
height = 480            # Height of Camera
frameR = 100            # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates
screen_width, screen_height = autopy.screen.size()
canvas = np.zeros((1280, 720, 3), dtype=np.uint8)
wScr, hScr = autopy.screen.size()
pyautogui.FAILSAFE = False
while True:
    ret, frame = co.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 1)

    frame = detector.locatefingers(frame)
    lmslist= detector.trackposition(frame)
    if len(lmslist) != 0:
        temp= lmslist[8][1:]
        temp2=lmslist[12][1:]
        x1, y1 = temp[0],temp[1]
        x2, y2 = temp2[0],temp2[1]
        # 3. Check which finger are up
        fingers = detector.findFingerUp()
        # 4. Only index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            autopy.mouse.move(screen_width - curr_x, curr_y)  # Moving the cursor
            cv2.circle(frame, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y
        # Both index and middle are up : left Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            length, frame, lineInfo = detector.findDistance(8, 12, frame)
            print(length)


            if length < 40:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

            # Both index and middle are up : Right Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            length, frame, lineInfo = detector.findDistance(4, 20, frame)
            print(length)

            if length < 40:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
    canvas = detector.draw_mode(frame, canvas)
    detector.screen_shot(frame)
    frame= cv2.GaussianBlur(frame,(5,5),20)
    cv2.imshow('Live Feed', frame)
    cv2.imshow('Drawing Frame ', canvas)
    if cv2.waitKey(1) == ord('q'):
        break

co.release()
cv2.destroyAllWindows()
