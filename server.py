from mlsocket import MLSocket
import torch
from facenet_pytorch import MTCNN
import time
import numpy as np


HOST = ''
PORT = 27004

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start = time.time()
mtcnn = MTCNN(keep_all=True, margin=0, device=device)
end = time.time()
print(f"loading mtcnn {device} took {round((end - start), 2)} seconds")


with MLSocket() as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"waiting for the client ...")
    conn, address = s.accept()
    print(f"connection estbalished")

    with conn:
        while True:
            try:
                img = conn.recv(1024)

                # note that the images can be batched.
                # img has to be in RGB format.
                boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

                if boxes is None:
                    boxes = np.array([])
                    probs = np.array([])
                    landmarks = np.array([])

                conn.send(boxes)
                conn.send(probs)
                conn.send(landmarks)
            except Exception as e:
                print(e)
                boxes = np.array([])
                probs = np.array([])
                landmarks = np.array([])

                conn.send(boxes)
                conn.send(probs)
                conn.send(landmarks)

    print("disconnected")
