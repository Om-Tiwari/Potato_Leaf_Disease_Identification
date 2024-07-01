import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Set page configuration
st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon=":potato:",
    layout="wide",
)

# Define the model architecture
def create_model():
    INPUT_SHAPE = (256, 256, 3)
    n_classes = 3
    model=tf.keras.Sequential([
        layers.Conv2D(32 , kernel_size = (3,3), activation='swish',input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, kernel_size=(3,3) , activation='swish'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, kernel_size=(3,3) , activation='swish'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3) , activation='swish'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3) , activation='swish'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3) , activation='swish'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation='swish'),
        layers.Dense(n_classes,activation='softmax'),
    ])
    return model

# Load the model weights
model = create_model()
model.load_weights("model_weights.h5")


# Define the class names for classification
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Define the function for making predictions
def classify_image(image):
    # Preprocess the image
    img = tf.image.resize(image, (256, 256))
    img = tf.expand_dims(img, axis=0)

    # Make a prediction
    pred = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(pred)]
    confidence = float(np.max(pred))

    # Return the predicted class and confidence score
    return {"class": predicted_class, "confidence": confidence}

# Set the title and description
st.title("Potato Leaf Disease Detector")
st.write("This app uses a deep learning model to detect potato leaf diseases.")


# Add a section for testing with example images
st.sidebar.subheader("Test with Example Images")

# Example image filenames
example_images = [
    "EarlyB.jpg",
    "LateB.png",
    "Healthy.jpg",
]

# Display example images
for example_image in example_images:
    st.sidebar.image(example_image, caption=example_image, use_column_width=True, width=250)

# Function to load and classify example images
def classify_example_image(example_image_path):
    image = Image.open(example_image_path)
    st.image(image, caption="Example Image", use_column_width=True)

    # Make a prediction
    prediction = classify_image(np.array(image))
    st.write(f"Predicted class: {prediction['class']}")
    st.write(f"Confidence: {prediction['confidence']*100:.2f}%")

# Add a selectbox for choosing example images
selected_example_image = st.sidebar.selectbox("Choose an example image", example_images)

# Classify the selected example image
if st.sidebar.button("Classify Example Image"):
    classify_example_image(selected_example_image)



# Add an image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    prediction = classify_image(np.array(image))
    st.write(f"Predicted class: {prediction['class']}")
    st.write(f"Confidence: {prediction['confidence']*100:.2f}%")

    # Add some additional details about the model
    st.write("---")
    st.write("**About the Model:**")
    st.write("This model was trained on a dataset of potato leaf images, including examples of early blight, late blight, and healthy leaves. The model architecture consists of several convolutional and pooling layers followed by dense layers. The model was trained using TensorFlow and Keras, and it achieved an accuracy of approximately 90% on a held-out test set.")

st.write("Made by https://www.linkedin.com/in/mrsus/")
