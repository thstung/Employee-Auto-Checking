import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import face_identification
import face_detection
import os
import cv2
import pandas as pd
import numpy as np
from settings import (
    DATA_FACE_DIR,
)


def main(name):
    df = pd.read_json(DATA_FACE_DIR)
    data_image = df["face"].to_numpy().tolist()
    members = df["name"].to_numpy().tolist()
    dir_image = os.listdir("Dataset/FaceData/raw/" + name)  # TODO dir of face

    for file in dir_image:
        image = cv2.imread("Dataset/FaceData/raw/" + name + "/" + file)
        image = face_detection.detect_face(image)[0][0]
        image_embedding = face_identification.embed_image(image)
        data_image.append(image_embedding)
        members.append(name)

    np.append(members, name)
    df = pd.DataFrame({"face": data_image, "name": members})
    df.to_json(DATA_FACE_DIR)

# dir = os.listdir('Dataset/FaceData/raw')
# for file in dir:
#     main(file)
if __name__ == "__main__":
    name = sys.argv[1]
    sys.exit(main(name) or 0)
