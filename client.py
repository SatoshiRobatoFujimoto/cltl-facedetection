from mlsocket import MLSocket
import numpy as np
import cv2
from contextlib import contextmanager
import time


HOST = ''
PORT = 27004


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_webcam_RGB():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yield img


# Send data
image_generator = yield_webcam_RGB()

with MLSocket() as s:

    s.connect((HOST, PORT))  # Connect to the port and host

    for img in image_generator:
        start = time.time()
        img_h, img_w, _ = np.shape(img)

        s.send(img)

        # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
        boxes = s.recv(1024)
        # This will also block until it receives all the data.
        probs = s.recv(1024)
        landmarks = s.recv(1024)  # Same

        for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            w = x2 - x1
            h = y2 - y1

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for (x, y) in landmark:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # img = cv2.resize(img, (img_w*2, img_h*2))
        cv2.imshow("result", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        end = time.time()
        print(f"{1/(end-start)} fps")

        if key == 27:  # ESC
            break
