import time

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class HandTracking:
    def __init__(
        self, min_detection_confidence=0.5, min_tracking_confidence=0.5, hand_limit=2
    ):
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            max_num_hands=hand_limit,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.is_drawing = False
        self.draw_points = []
        self.last_drawn_point = None
        self.current_points = []
        self.last_screen_shot = float(0)
        self.last_drawmode_toggle = float(0)
        self.current_time = time.time()
        self.drawing_window_open = False

    def locate_fingers(self, frame):
        convt_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(convt_image)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.draw.draw_landmarks(frame, hand, self.handsMp.HAND_CONNECTIONS)
        return frame


frame = None
gesture = None
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
    return jsonify({"status": gesture})


def camera_is_connected():
    return True


def camera_is_connecting():
    return False


def generate_frames():
    global frame
    while True:
        if frame is not None:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            resized_frame = cv2.resize(frame, (640, 480))

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
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    global frame
    vc = cv2.VideoCapture(0)

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

            frame = detector.locate_fingers(flipped_frame)

            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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
