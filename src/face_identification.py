import cv2
import numpy as np
from openvino.inference_engine import IECore
from settings import (
    MODEL_IDENTIFY_XML,
    MODEL_IDENTIFY_BIN,
    FACE_SIZE,
)

# Load the Inference Engine API
ie = IECore()

# Load the ReID model
model_xml = MODEL_IDENTIFY_XML
model_bin = MODEL_IDENTIFY_BIN
net = ie.read_network(model=model_xml, weights=model_bin)

# Load the model into the inference engine
exec_net = ie.load_network(network=net, device_name="CPU")
input_blob = next(iter(net.input_info))


# Load an image and resize it
def embed_image(image: np.ndarray):
    image = cv2.resize(image, (FACE_SIZE, FACE_SIZE))

    # Prepare the image for inference
    image = image.transpose((2, 0, 1))
    input_data = np.expand_dims(image, axis=0)

    # Run inference on the image
    output = exec_net.infer(inputs={input_blob: image})
    output_blob = next(iter(output))
    output_data = output[output_blob][0]
    return output_data.flatten()


def calculate_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def result_name(image, data_train, classes, threshold=0.75):
    image_embedding = embed_image(image)
    for i in range(len(classes)):
        cosine = calculate_cosine_similarity(image_embedding, data_train[i])
        if abs(cosine) >= threshold:
            return classes[i]
    return "Unknown"
