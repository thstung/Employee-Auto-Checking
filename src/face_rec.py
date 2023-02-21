import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
import time
import face_detection
import face_identification
import cv2
import pandas as pd
from settings import (
    DATA_FACE_DIR,
)

df = pd.read_json(DATA_FACE_DIR)
data_faces = df["face"].to_numpy().tolist()
members = df["name"].to_numpy().tolist()


def main(image):
    imgs, x, y = face_detection.detect_face(image)
    if len(imgs) == 0:
        return None
    for i in range(len(imgs)):
        xmin, xmax = x[i]
        ymin, ymax = y[i]
        bbox = [[xmin, ymin], [xmax, ymax]]
        frame = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        name_member = face_identification.result_name(imgs[i], data_faces, members)
        if name_member != "Unknown":
            current_time = datetime.now()
            break
    return frame, bbox, name_member, current_time
# if __name__ == "__main__":
#     main(image)
