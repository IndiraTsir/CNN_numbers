import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import plotly.express as px


st.set_page_config(
    page_title="Digit Recognition App",
    layout="wide")

st.title("Digit Recognition App")

st.write('### Draw a digit in 0-9 in the box below')
# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

#==============================================================
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee") # background color hex
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"]) # drag and drop file
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
# ) # drawing tool forms etc
#==============================================================

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
#     background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
)

# Load the model
model = load_model('model_Indi_2.h5')
# model = tf.keras.models.load_model("my_model2.h5")

# Load the test dataset
Test = ('data/test.csv')

#==============================================================

# # Convert the 'path' column to bytes
# df['data/test.csv'] = Test['data/test.csv'].astype(bytes)
#==============================================================



# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    image = canvas_result.image_data
    image1 = image.copy()
    image1 = image1.astype('uint8')
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1,(28,28))
    st.image(image1)

    image1.resize(1,28,28,1)
    st.title(np.argmax(model.predict(image1)))
if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    
st.write('### Prediction') 
    
# # Define a function to display a random image from the test dataset
# def get_random_image():
#     index = np.random.choice(X_test.shape[0])
#     image = X_test[index]
#     true_label = y_test[index]
#     return image, true_label

# # Define a function to predict the label of an image
# def predict_label(image):
#     image = image / 255.0  # Normalize the pixel values
#     image = np.expand_dims(image, axis=0)  # Add a batch dimension
#     pred = model.predict(image)
#     pred_label = np.argmax(pred, axis=1)[0]
#     return pred_label

# # Define the Streamlit app
# def app():
#     st.title('Digit Recognition')
    
#     # Get a random image and display it
#     image, true_label = get_random_image()
#     st.image(image, caption=f'True label: {true_label}', width=200)
    
#     # Make a prediction on the image
#     if st.button('Predict'):
#         pred_label = predict_label(image)
#         st.write(f'Predicted label: {pred_label}')
        
#         # Allow the user to validate the prediction
#         if st.button('Correct'):
#             st.write('Great!')
#         if st.button('Incorrect'):
#             st.write('Oops!')

# if __name__ == '__main__':
#     app()
