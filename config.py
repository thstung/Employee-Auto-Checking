from streamlit_webrtc import RTCConfiguration

# MODELS
# MODEL_PATH = "models"
MODEL_CONFIG = "models/deploy.prototxt"
MODEL_FACE_DETECTOR = "models/res10_300x300_ssd_iter_140000.caffemodel"
MODEL_MASK_DETECTOR = "models/mask_detector.model"

# Config server for webcam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# IMAGE
IMAGE_EXAMPLE = "dataset/images/out.jpeg"
IMAGE_TYPES = ["jpg", "jpeg", "png"]