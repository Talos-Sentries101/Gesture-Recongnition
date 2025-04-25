import io

import autopy
import comtypes
import numpy as np
import mediapipe as mp
import cv2
import math as math
from ctypes import cast, POINTER
import time
import pyautogui
import win32clipboard
from PIL import ImageGrab
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbcontrol

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
range_vol=volume.GetVolumeRange()
minVol, maxVol =    range_vol[0], range_vol[1]
tx_old = 0
ty_old = 0
trial = True
flag = False
grabflag = False
pinchmajorflag = False
pinchminorflag = False
pinchstartxcoord = None
pinchstartycoord = None
pinchdirectionflag = None
prevpinchlv = 0
pinchlv = 0
prev_hand = None
pinch_threshold = 0.3
def is_muted():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    return volume.GetMute()


class Handtracking:
    def __init__(self, mode=False, limit_hands=1, detection=0.7, tracktion=0.7):
        self.__mode__ = mode # Mode to toggle to different modes
        self.__maxHands__ = limit_hands #max no of hands detected
        self.__detectionCon__ = detection #detection factor
        self.__trackCon__ = tracktion # tracking factor
        self.handsMp = mp.solutions.hands # hand object
        self.hands = self.handsMp.Hands(static_image_mode=mode,
            max_num_hands=limit_hands,
            min_detection_confidence=detection,
            min_tracking_confidence=tracktion)
        self.mpDraw = mp.solutions.drawing_utils #drawing hands
        self.tipIds = [4, 8, 12, 16, 20] # tips of all the fingers in id form as appended in lmslist[id of lm,x-xords,y-cords,0]
        #self.on_or_off = 0
        self.is_drawing = False
        self.draw_points =[]
        self.last_drawn_point = None #last drawn point for drawing mode
        self.current_points=[] #Current point tracking for drawing mode
        self.last_drawmode_toggle = False #draw mode toogle
        self.current_time = time.time() # For starting a clock to toogle input lag for holding gestures
        self.drawing_window_open = False #Flag for drawing window on or off
        self.last_screen_shot = float(0) # SS feature
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
               cz=int(lm.z)
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
            if self.lmslist[self.tipIds[id]][2] < self.lmslist[self.tipIds[id] - 1][2]:
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
            vol = np.interp(length, [30, 350], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

    def draw_mode(self, frame, canvas, fingers=None):
        if fingers is None or not self.lmslist:
            return canvas

        h, w, _ = frame.shape


        if self.is_drawing and fingers == [0, 0, 1, 1, 1]:
            x, y = self.lmslist[8][1], self.lmslist[8][2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            if self.last_drawn_point is None or math.hypot(x - self.last_drawn_point[0],
                                                           y - self.last_drawn_point[1]) > 5:
                self.current_points.append((x, y))
                self.last_drawn_point = (x, y)
        else:
            if self.is_drawing and self.current_points:
                self.draw_points.append(self.current_points)
                self.current_points = []
            self.last_drawn_point = None

        for points in self.draw_points:
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], (0, 0, 255), 10)
        for i in range(1, len(self.current_points)):
            cv2.line(canvas, self.current_points[i - 1], self.current_points[i], (0, 0, 255), 10)

        if fingers == [0, 0, 0, 0, 1]:
            print('Clearing canvas')
            self.draw_points = []
            self.current_points = []
            self.last_drawn_point = None
            canvas.fill(0)


        return canvas

    def screen_shot(self, frame):
        if not self.lmslist:
            return
        fingers = self.findFingerUp()
        current_time =time.time()
        if fingers == [1, 0, 1, 1, 1] :
            if int(abs(current_time - self.last_screen_shot)) > 2:
                img = ImageGrab.grab()
                self.last_screen_shot = current_time

                #copy to clipboard
                output = io.BytesIO()
                img.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # BMP file header starts from 14th byte
                output.close()

                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                print("Screenshot copied to clipboard!")



    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        l1=self.lmslist[p1][1:]
        l2= self.lmslist[p2][1:]
        x1, y1 = l1[0],l1[1]
        x2, y2 = l2[0],l2[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        #if draw:
            #cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            #cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            #cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            #cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]
    def Mousecontrol(self,frame):
        frameR = 100  # frame Reduction
        smoothening = 7
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        p1, q1 = self.lmslist[8][1:]
        p2, q2 = self.lmslist[12][1:]
        fingers= self.findFingerUp()
        wScr, hScr = autopy.screen.size()
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(p1, (frameR, 640 - frameR), (0, wScr))
            y3 = np.interp(q1, (frameR, 480 - frameR), (0, hScr))

            # 6. Smoothen Valuse
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        return frame

    def get_position(self):
        """
        returns coordinates of current hand position.

        Locates hand to get cursor position also stabilize cursor by
        dampening jerky motion of hand.

        Returns
        -------
        tuple(float, float)
        """
        point = 9
        position = [self.lmslist[point][point][0], self.lmslist[point][1]]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if prev_hand is None:
            prev_hand = x, y
        delta_x = x - prev_hand[0]
        delta_y = y - prev_hand[1]
import io

import autopy
import comtypes
import numpy as np
import mediapipe as mp
import cv2
import math as math
from ctypes import cast, POINTER
import time
import pyautogui
import win32clipboard
from PIL import ImageGrab
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbcontrol

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
range_vol=volume.GetVolumeRange()
minVol, maxVol =    range_vol[0], range_vol[1]
tx_old = 0
ty_old = 0
trial = True
flag = False
grabflag = False
pinchmajorflag = False
pinchminorflag = False
pinchstartxcoord = None
pinchstartycoord = None
pinchdirectionflag = None
prevpinchlv = 0
pinchlv = 0
prev_hand = None
pinch_threshold = 0.3
def is_muted():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    return volume.GetMute()


class Handtracking:
    def __init__(self, mode=False, limit_hands=1, detection=0.7, tracktion=0.7):
        self.__mode__ = mode # Mode to toggle to different modes
        self.__maxHands__ = limit_hands #max no of hands detected
        self.__detectionCon__ = detection #detection factor
        self.__trackCon__ = tracktion # tracking factor
        self.handsMp = mp.solutions.hands # hand object
        self.hands = self.handsMp.Hands(static_image_mode=mode,
            max_num_hands=limit_hands,
            min_detection_confidence=detection,
            min_tracking_confidence=tracktion)
        self.mpDraw = mp.solutions.drawing_utils #drawing hands
        self.tipIds = [4, 8, 12, 16, 20] # tips of all the fingers in id form as appended in lmslist[id of lm,x-xords,y-cords,0]
        #self.on_or_off = 0
        self.is_drawing = False
        self.draw_points =[]
        self.last_drawn_point = None #last drawn point for drawing mode
        self.current_points=[] #Current point tracking for drawing mode
        self.last_drawmode_toggle = False #draw mode toogle
        self.current_time = time.time() # For starting a clock to toogle input lag for holding gestures
        self.drawing_window_open = False #Flag for drawing window on or off
        self.last_screen_shot = float(0) # SS feature
        self.plocX, self.plocY = 0, 0# Used in smoothing the drawing feature(Past cords x and y)
        self.clocX, self.clocY = 0, 0# Used in smoothing the drawing feature(Current cords x and y)
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
               cz=int(lm.z)
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
            if self.lmslist[self.tipIds[id]][2] < self.lmslist[self.tipIds[id] - 1][2]:
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
            vol = np.interp(length, [30, 350], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

    def draw_mode(self, frame, canvas, fingers=None):
        if fingers is None or not self.lmslist:
            return canvas

        h, w, _ = frame.shape


        if self.is_drawing and fingers == [0, 0, 1, 1, 1]:
            x, y = self.lmslist[8][1], self.lmslist[8][2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            if self.last_drawn_point is None or math.hypot(x - self.last_drawn_point[0],
                                                           y - self.last_drawn_point[1]) > 5:
                self.current_points.append((x, y))
                self.last_drawn_point = (x, y)
        else:
            if self.is_drawing and self.current_points:
                self.draw_points.append(self.current_points)
                self.current_points = []
            self.last_drawn_point = None

        for points in self.draw_points:
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], (0, 0, 255), 10)
        for i in range(1, len(self.current_points)):
            cv2.line(canvas, self.current_points[i - 1], self.current_points[i], (0, 0, 255), 10)

        if fingers == [0, 0, 0, 0, 1]:
            print('Clearing canvas')
            self.draw_points = []
            self.current_points = []
            self.last_drawn_point = None
            canvas.fill(0)


        return canvas

    def screen_shot(self, frame):
        if not self.lmslist:
            return
        fingers = self.findFingerUp()
        current_time =time.time()
        if fingers == [1, 0, 1, 1, 1] :
            if int(abs(current_time - self.last_screen_shot)) > 2:
                img = ImageGrab.grab()
                self.last_screen_shot = current_time

                #copy to clipboard
                output = io.BytesIO()
                img.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # BMP file header starts from 14th byte
                output.close()

                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
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
    def Mousecontrol(self,frame):
        frameR = 100  # frame Reduction
        smoothening = 7
        p1, q1 = self.lmslist[8][1:]
        p2, q2 = self.lmslist[12][1:]
        fingers= self.findFingerUp()
        wScr, hScr = autopy.screen.size()
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(p1, (frameR, 640 - frameR), (0, wScr))
            y3 = np.interp(q1, (frameR, 480 - frameR), (0, hScr))

            # 6. Smoothen Valuse
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        return frame

    def get_position(self):
        """
        returns coordinates of current hand position.

        Locates hand to get cursor position also stabilize cursor by
        dampening jerky motion of hand.

        Returns
        -------
        tuple(float, float)
        """
        point = 9
        position = [self.lmslist[point][point][0], self.lmslist[point][1]]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if prev_hand is None:
            prev_hand = x, y
        delta_x = x - prev_hand[0]
        delta_y = y - prev_hand[1]

        distsq = delta_x ** 2 + delta_y ** 2
        ratio = 1
        prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1 / 2))
        else:
            ratio = 2.1
        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)
    def getpinchylv(self):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((pinchstartycoord - self.lmslist[8][1]) * 10, 1)
        return dist

    def getpinchxlv(self):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((self.lmslist[8][0] - Controller.pinchstartxcoord) * 10, 1)
        return dist

co=cv2.VideoCapture(0)
co.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
co.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = Handtracking()
canvas = np.zeros((1280, 720, 3), dtype=np.uint8)

if not co.isOpened():
    print("Error accesing camera plz check for any access related issue or a camera shutter blocking ")
    exit()
width = 1280             # Width of Camera
height = 720            # Height of Camera
frameR = 60          # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates
screen_width, screen_height = autopy.screen.size()
canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
draw_toggle_gesture= [0,1,0,0,1]
wScr, hScr = autopy.screen.size()
pyautogui.FAILSAFE = False
prev_positions = []
threshold = 5
max_clicks = 3
while True:
    ret, frame = co.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 1)

    frame = detector.locatefingers(frame)
    lmslist= detector.trackposition(frame)
    fingers = detector.findFingerUp()
    current_time = time.time()
    if fingers == draw_toggle_gesture:
        if abs(current_time - detector.last_drawmode_toggle) > 5:
            detector.is_drawing = not detector.is_drawing
            print(f"Draw mode toggled: {detector.is_drawing}")
            detector.last_drawmode_toggle = current_time

    canvas = detector.draw_mode(frame, canvas, fingers)

    if detector.is_drawing and abs(current_time - detector.last_drawmode_toggle) > 1:

        cv2.imshow('Drawing Frame', canvas)

        cv2.putText(frame, "DRAW MODE ON", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        detector.drawing_window_open = True
    else:
        if detector.drawing_window_open:
            cv2.destroyWindow('Drawing Frame')
            detector.drawing_window_open = False
    if detector.is_drawing is False:
        if len(lmslist) != 0:
            temp= lmslist[8][1:]
            temp2=lmslist[12][1:]
            x1, y1 = temp[0],temp[1]
            x2, y2 = temp2[0],temp2[1]
            # 3. Check which finger are up
            fingers = detector.findFingerUp()
            # 4. Only index finger: Moving mode
            if fingers[0]==1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3]==0 and fingers[4]==0:
                x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening
                #pyautogui.moveTo(curr_x, curr_y)

                autopy.mouse.move(curr_x, curr_y)  # Moving the cursor
                cv2.circle(frame, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y
            if fingers== [1,0,0,0,0]:
                pyautogui.scroll(-60)
                time.sleep(0.2)
            if fingers == [1,1,1,1,1]:
                pyautogui.scroll(60)
                time.sleep(0.2)
            # Both index and middle are up : left Clicking mode
            if fingers[1] == 1 and fingers[2] == 1:

                length, frame, lineInfo = detector.findDistance(8, 12, frame)


                if length < 40:
                    cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()


                # Both index and middle are up : Right Clicking mode
            if fingers[1] == 1 and fingers[2] == 1:
                length, frame, lineInfo = detector.findDistance(4, 20, frame)
                if length < 20 :
                    #cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
            if fingers[0]== 0 and fingers[2]==1 and fingers[1]==0:
                tolarence=time.time()
                length, frame, lineInfo = detector.findDistance(4, 12, frame)
                cv2.circle(frame,(lineInfo[4],lineInfo[5]),18,(0,255,0),cv2.FILLED)
                vol = np.interp(int(length), [30, 300], [minVol, maxVol])
                if is_muted():
                    volume.SetMute(0, None)
                if tolarence > threshold:
                    #if length <
                    volume.SetMasterVolumeLevel(vol, None)
    detector.current_time= current_time
    detector.screen_shot(frame)
    frame= cv2.GaussianBlur(frame,(5,5),20)
    cv2.imshow('Live Feed', frame)

    if cv2.waitKey(1) == ord('q'):
        break

co.release()
cv2.destroyAllWindows()
        distsq = delta_x ** 2 + delta_y ** 2
        ratio = 1
        prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1 / 2))
        else:
            ratio = 2.1
        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)
    def getpinchylv(self):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((pinchstartycoord - self.lmslist[8][1]) * 10, 1)
        return dist

    def getpinchxlv(self):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((self.lmslist[8][0] - Controller.pinchstartxcoord) * 10, 1)
        return dist

co=cv2.VideoCapture(0)
co.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
co.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = Handtracking()
canvas = np.zeros((1280, 720, 3), dtype=np.uint8)

if not co.isOpened():
    print("Error accesing camera plz check for any access related issue or a camera shutter blocking ")
    exit()
width = 1280             # Width of Camera
height = 720            # Height of Camera
frameR = 60          # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates
screen_width, screen_height = autopy.screen.size()
canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
draw_toggle_gesture= [0,1,0,0,1]
wScr, hScr = autopy.screen.size()
pyautogui.FAILSAFE = False
prev_positions = []
threshold = 5
max_clicks = 3
while True:
    ret, frame = co.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 1)

    frame = detector.locatefingers(frame)
    lmslist= detector.trackposition(frame)
    fingers = detector.findFingerUp()
    current_time = time.time()
    if fingers == draw_toggle_gesture:
        if abs(current_time - detector.last_drawmode_toggle) > 5:
            detector.is_drawing = not detector.is_drawing
            print(f"Draw mode toggled: {detector.is_drawing}")
            detector.last_drawmode_toggle = current_time

    canvas = detector.draw_mode(frame, canvas, fingers)

    if detector.is_drawing and abs(current_time - detector.last_drawmode_toggle) > 1:

        cv2.imshow('Drawing Frame', canvas)

        cv2.putText(frame, "DRAW MODE ON", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        detector.drawing_window_open = True
    else:
        if detector.drawing_window_open:
            cv2.destroyWindow('Drawing Frame')
            detector.drawing_window_open = False


    if len(lmslist) != 0:
        temp= lmslist[8][1:]
        temp2=lmslist[12][1:]
        x1, y1 = temp[0],temp[1]
        x2, y2 = temp2[0],temp2[1]
        # 3. Check which finger are up
        fingers = detector.findFingerUp()
        # 4. Only index finger: Moving mode
        if fingers[0]==1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3]==0 and fingers[4]==0:
            x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening
            #pyautogui.moveTo(curr_x, curr_y)

            autopy.mouse.move(curr_x, curr_y)  # Moving the cursor
            cv2.circle(frame, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y
        if fingers== [1,0,0,0,0]:
            pyautogui.scroll(-60)
            time.sleep(0.2)
        if fingers == [1,1,1,1,1]:
            pyautogui.scroll(60)
            time.sleep(0.2)
        # Both index and middle are up : left Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            length, frame, lineInfo = detector.findDistance(8, 12, frame)


            if length < 40:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


            # Both index and middle are up : Right Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, frame, lineInfo = detector.findDistance(4, 20, frame)
            if length < 20 :
                #cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
        if fingers[0]== 0 and fingers[2]==1 and fingers[1]==0:
           tolarence=time.time()
           length, frame, lineInfo = detector.findDistance(4, 12, frame)
           cv2.circle(frame,(lineInfo[4],lineInfo[5]),18,(0,255,0),cv2.FILLED)
           vol = np.interp(int(length), [30, 300], [minVol, maxVol])
           if is_muted():
               volume.SetMute(0, None)
           if tolarence > threshold:
               #if length <
               volume.SetMasterVolumeLevel(vol, None)
    detector.current_time= current_time
    detector.screen_shot(frame)
    frame= cv2.GaussianBlur(frame,(5,5),20)
    cv2.imshow('Live Feed', frame)

    if cv2.waitKey(1) == ord('q'):
        break

co.release()
cv2.destroyAllWindows()
