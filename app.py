import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Setup the Page Title
st.set_page_config(page_title="Face Mask Detector")
st.title("😷 Face Mask Detection Web App")
st.write("Upload a photo to see if people are wearing masks correctly.")

# 2. Load the AI Model (Make sure 'best.pt' is in the same folder!)
try:
    model = YOLO('best.pt') 
except Exception as e:
    st.error("Error: 'best.pt' not found. Please put your trained model file in this folder.")

# 3. Create the File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Show a "Processing" message
    with st.spinner('AI is analyzing...'):
        # Convert image to a format the AI understands
        img_array = np.array(image)
        
        # Run the AI Detection
        results = model(img_array)
        
        # Plot the boxes/labels on the image
        res_plotted = results[0].plot()
        
        # Display the result
        st.subheader("Results:")
        st.image(res_plotted, channels="BGR", use_container_width=True)
        
        # Logic to tell the user the result in text
        names = model.names
        for c in results[0].boxes.cls:
            st.info(f"Detected: {names[int(c)]}")