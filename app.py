import streamlit as st
import cv2
from code.misc.feature_extractor.feature_extractor import YouTube8MFeatureExtractor
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


loaded_model = load_model("models/new_model.h5")

# Reference: https://github.com/google/youtube-8m/tree/master/feature_extractor
# Note that the original module only works for TF 1.x but not 2.0,
# we need to modify the script specifically for tf.GraphDef, tf.Graph, and tf.Session to be
# tf.compat.v1.GraphDef tf.compat.v1.Graph and tf.compat.v1.Session, respectively.

CAP_PROP_POS_MSEC = 0

def frame_iterator(filename, every_ms=1000, max_num_frames=300):
    video_capture = cv2.VideoCapture()
    if not video_capture.open(filename):
        return
    last_ts = -99999  # The timestamp of last retrieved frame.
    num_retrieved = 0
    while num_retrieved < max_num_frames:
        # Skip frames
        while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
            if not video_capture.read()[0]:
                return
        last_ts = video_capture.get(CAP_PROP_POS_MSEC)
        has_frames, frame = video_capture.read()
        if not has_frames:
            break
        yield frame
        num_retrieved += 1

# Pre-trained ImageNet Inception model and PCA matrices will be downloaded if not found.
extractor = YouTube8MFeatureExtractor("code/misc/models")


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
  """Quantizes float32 ⁠ features ⁠ into string."""
  assert features.dtype == 'float32'
  assert len(features.shape) == 1  # 1-D array
  features = np.clip(features, min_quantized_value, max_quantized_value)
  quantize_range = max_quantized_value - min_quantized_value
  features = (features - min_quantized_value) * (255.0 / quantize_range)
  features = [int(round(f)) for f in features]

  return features


video_file = "./data/Test_Video/Music.mp4"  # A test sample.

def predict(video_file):
    rgb_features = []
    sum_rgb_features = None

    fiter = frame_iterator(video_file, every_ms=1000.0)

    max_frames = 45  # Number of frames to extract

    for _ in range(max_frames):
        frame = next(fiter, None)
        if frame is None:
            break
        features = extractor.extract_rgb_frame_features(frame[:, :, ::-1])
        features = quantize(features)
        rgb_features.append(features)


    # Convert the list to a numpy array
    rgb_features_array = np.array(rgb_features)

    # Reshape the array to add a new axis at the beginning
    rgb_features_array = np.expand_dims(rgb_features_array, axis=0)

    predictions = loaded_model.predict(rgb_features_array)

    return predictions


# streamlit app where user can upload a video file and get the predictions
# the user can also choose a video already in the data folder


st.title("What videos do you love to take? 🎥 - Maxime Wolf")

st.write("Upload a video file to get predictions")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
predictions = None
st.write("Or choose a video from the data folder: ")
# let user select title
video_titles = os.listdir("data/Test_Video")
video_title = st.selectbox("Choose a video", [""] + video_titles)

if video_title:
    st.video(f"data/Test_Video/{video_title}")
    video_file = f"data/Test_Video/{video_title}"
    with st.spinner("Predicting..."):
      predictions = predict(video_file)

if uploaded_file is not None:
    st.video(uploaded_file)
    video_file = uploaded_file
    with st.spinner("Predicting..."):
        predictions = predict(video_file)



if predictions is not None:
    categories = ["Vehicle", "Concert", "Dance", "Football", "Animal", "Food", "Outdoor recreation", "Nature", "Mobile phone", "Cooking"]

    st.markdown("### Predictions")
    st.markdown("The category for your video is: **{}**".format(categories[np.argmax(predictions)]))

    dict = zip(categories, predictions[0] * 100)
    # histogram with prediction for each category
    st.bar_chart(data = pd.DataFrame(dict, columns=["Category", "Prediction"]), x="Category", y="Prediction (%)")







if __name__ == "__main__":
    predictions = predict(video_file)