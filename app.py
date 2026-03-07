import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Sushi Detection AI",
    page_icon="🍣",
    layout="wide"
)

# -------------------------
# Title
# -------------------------
st.title("🍣 Sushi Detection AI")
st.markdown("Upload a video and run the trained YOLO model to detect sushi types.")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("runs/detect/train9/weights/best.pt", device='0')
    return model

model = load_model()

# -------------------------
# Upload Video
# -------------------------
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"]
)

# -------------------------
# Run button
# -------------------------
if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🚀 Run Detection"):

        st.write("Processing video...")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO detection
            results = model(frame)

            annotated_frame = results[0].plot()

            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        st.success("Detection completed!")
