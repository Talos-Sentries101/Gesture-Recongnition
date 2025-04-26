import math
import time
from collections import deque

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify
from flask_cors import CORS
from pynput.mouse import Controller

app = Flask(__name__)
CORS(app)


class HandTracking:
    def __init__(
        self, min_detection_confidence=0.5, min_tracking_confidence=0.5, hand_limit=2
    ):
        self.detectionCon = min_detection_confidence
        self.trackingCon = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hand_limit = hand_limit
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.detectionCon,
            max_num_hands=self.hand_limit,
            min_tracking_confidence=self.trackingCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.is_drawing = False
        self.draw_points = []
        self.last_drawn_point = None
        self.current_points = []
        self.last_screen_shot = float(0)
        self.last_drawmode_toggle = float(0)
        self.current_time = time.time()
        self.drawing_window_open = False

    def find_hands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return frame

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
                )

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if len(self.lmList) == 0:
            return []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def throttle(ms_interval):
    def decorator(func):
        last_called = [0]

        def wrapper(*args, **kwargs):
            now = time.time() * 1000
            if now - last_called[0] >= ms_interval:
                last_called[0] = int(now)
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


final_frame = None
draw_frame = None
gesture = None
mode = False
detector = HandTracking()


@app.route("/camera_status")
def camera_status():
    if camera_is_connected():
        return jsonify({"status": "connected"})
    elif camera_is_connecting():
        return jsonify({"status": "connecting"})
    else:
        return jsonify({"status": "idle"})


@app.route("/gesture_status")
def gesture_status():
    return jsonify({"status": gesture, "mode": mode})


def camera_is_connected():
    return True


def camera_is_connecting():
    return False


def encode_frames_for_http(final_frame):
    while True:
        if final_frame is not None:
            if len(final_frame.shape) == 2:
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_GRAY2BGR)

            resized_frame = cv2.resize(final_frame, (640, 480))

            ret, jpeg = cv2.imencode(".jpg", resized_frame)
            if not ret:
                break

            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
            )


@app.route("/video_feed")
def video_feed():
    global final_frame
    return Response(
        encode_frames_for_http(final_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/draw_feed")
def draw_feed():
    global draw_frame
    return Response(
        encode_frames_for_http(draw_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@throttle(10)
def move(x, y, mouse):
    mouse.position = (x, y)


def smoothen(cx, cy, px, py, sf, mt):
    smoothed_x = px + (cx - px) * sf
    smoothed_y = py + (cy - py) * sf

    dist = math.hypot(smoothed_x - px, smoothed_y - py)
    if dist > mt:
        return smoothed_x, smoothed_y
    else:
        return None, None


def main():
    global final_frame, draw_frame, gesture, mode
    vc = cv2.VideoCapture(0)
    mouse = Controller()

    cursor_gesture = [0, 0, 1, 1, 1]

    wVid, hVid = 640, 480
    frameRate = 100
    smoothF = 0.6
    move_threshold = 13
    px, py = 0, 0

    pTime = 0

    dist = deque(maxlen=10)

    vc.set(3, wVid)
    vc.set(4, hVid)

    if not vc.isOpened():
        print("Error: Unable to open video stream")
        return

    try:
        while True:
            ret, unflipped_frame = vc.read()
            if not ret:
                print("Failed to read frame")
                break

            flipped_frame = cv2.flip(unflipped_frame, 1)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(
                flipped_frame,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )

            final_frame = detector.find_hands(flipped_frame)
            lmList, bbox = detector.findPosition(flipped_frame)
            fingers = detector.fingersUp()

            if len(lmList) != 0:
                if fingers == cursor_gesture:
                    x1, y1 = lmList[4][1:]
                    x2, y2 = lmList[8][1:]
                    curr_dist = distance(x1, y1, x2, y2)
                    dist.append(curr_dist)

                    print(sum(dist) / 10)
                    if len(dist) == 10 and sum(dist) / 10 < 25:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cx, cy = smoothen(cx, cy, px, py, smoothF, move_threshold)
                        if cx is not None and cy is not None:
                            px, py = cx, cy
                            move(cx, cy, mouse)

            cv2.imshow("Video", final_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                vc.release()
                cv2.destroyAllWindows()
                exit()
    finally:
        vc.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from threading import Thread

    flask_thread = Thread(
        target=app.run, kwargs={"debug": True, "threaded": True, "use_reloader": False}
    )
    flask_thread.start()

    main()
