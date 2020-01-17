import numpy as np
import streamlit as st
from fastai import *
from fastai.vision import *
from RAdam import RAdam
import PIL 

# Function to Predict Planets
def predict(img):
    # Initialize Directories
    model_dir = Path("./models/")
    # Load the CNN Model
    learner = load_learner(model_dir)
    # Return the Predictions
    return str(learner.predict(img)[0]).capitalize()


# User Interface
st.title("Planet Recognizer AI")
st.markdown(">Hi there, this is a **machine-learning powered app** which can tell you which of the 8 planets your image has. Give it a try. It is fun!")
st.markdown("Built by [Abhinand](https://www.linkedin.com/in/abhinand-05/)")
st.write("")
st.markdown("**Note:** Your uploaded image should contain atleast one planet. Since this is being treated as classification task and not object detection, it can detect only 1 planet.")
st.markdown("*This app is in it's early stages!* Will be updated regularly.")
st.write("")
uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))
if uploaded_file is not None:
    image = open_image(uploaded_file)
    st.image(uploaded_file, caption='Uploaded Image.', width=360)
    st.markdown("Hurray, You got the AI thinking!")
    st.write("")
    st.markdown("**AI**: I think this is...")
    label = predict(image)
    st.success(label)

st.markdown("")
st.markdown("[GitHub Repo](https://github.com/abhinand5/planets-recognizer-app)")