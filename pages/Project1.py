import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
import io
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

html_start = """
                <div style="background: rgba(255, 255, 255, 0.2);
                        border-radius: 12px;
                        box-shadow: 0 4px 30px rgba(80, 50, 220, 0.4);
                        backdrop-filter: blur(5px);
                        -webkit-backdrop-filter: blur(5px);
                        border: 1px solid rgba(255, 255, 255, 0.2);"> 
                    <h2 style="color:white;text-align:center;font-size:24px">Image Classification</h2>
                </div>
                <body>
                <br>
                    Project Description:
                    <p>
                    The goal of this project is to develop an image classification system that can accurately classify images of objects into three classes: flowers, car wheels, and muffins. The images used for training and testing the model are clicked personally from a mobile phone, ensuring that the dataset is representative of real-world scenarios.
                    </p>
                    <p>
                    The first step in the project is to preprocess the image data. This includes resizing the images to a consistent size, normalizing pixel values, and splitting the dataset into training and validation sets to evaluate the performance of the model.
                    </p>
                    <p>
                    To improve the model's ability to generalize to new images, image augmentation techniques are employed. This involves applying random transformations to the images during training, such as rotation, flipping, and zooming, to create additional training examples with slight variations.
                    </p>
                    <p>
                    Two different approaches are used to build the image classification system. First, a pre-trained model is utilized. Transfer learning is leveraged to leverage the knowledge learned from a large dataset to improve the model's performance on the smaller dataset of personally clicked images. The pre-trained model is fine-tuned on the dataset, and its performance is evaluated.
                    </p>
                    <p>
                    Second, a custom neural network model is developed. The architecture of the neural network is designed from scratch, tailored to the characteristics of the image dataset. The model is trained using the preprocessed and augmented dataset, and its performance is compared with the pre-trained model.
                    </p>
                    <b>Challenges Faced:</b>
                    <p>
                    One challenge faced during the project was that the validation accuracy score was consistently higher than the training accuracy. To mitigate this, an analysis was conducted to identify potential causes, such as data leakage or overfitting. After thorough investigation, a simple solution was implemented to address the issue, which included adjusting the learning rate and regularization techniques.
                    </p>
                    <b>Conclusion:</b>
                    <p>
                    In conclusion, this project aimed to develop an image classification system to classify images of flowers, car wheels, and muffins. The dataset used for training and testing was created from personally clicked images from a mobile phone. Various techniques were employed, including image preprocessing, augmentation, and utilization of both pre-trained and custom neural network models. Challenges were encountered, but with careful analysis and adjustments, a satisfactory solution was achieved. The developed model can accurately classify the objects of interest and can be further improved with additional data and fine-tuning.
                    </p>
                </body>
            """     
st.markdown(html_start, unsafe_allow_html=True)
html_demo = """
            <div>
                <h2 style="color:white;text-align:center;font-size:20px">To test the model download one of the images and load the same below</h2>
            </div>
            """
st.markdown(html_demo, unsafe_allow_html=True)
img_html = """
            <div>
                <a download="flower.jpg" href="flower.jpg" title="Flower">
                    <img alt="Flower" src="../flower.jpg">
                </a>
                <a download="muffin.jpg" href="muffin.jpg" title="Muffin">
                    <img alt="Muffin" src="../muffin.jpg">
                </a>
                <a download="wheel.jpg" href="wheel.jpg" title="MuffWheelin">
                    <img alt="Wheel" src="../muffin.jpg">
                </a>
            </div>
            """
st.markdown(img_html, unsafe_allow_html=True)

img = st.file_uploader("Choose a file")
if img is not None:
    # Convert the uploaded file to a PIL image
    pil_img = Image.open(io.BytesIO(img.read()))
    pil_img = pil_img.resize((224, 224))
    img_array = np.array(pil_img)
    img_array = img_array / 255.0 # Rescale pixel values to [0, 1]

    # Expand dimensions to match the batch size expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Create a dummy label since the model expects a batch of images with labels
    label = np.array([0]) # change label as per your requirement

    # Create a generator with the single image and dummy label
    test_data_generator = ImageDataGenerator().flow(
        x=img_array,
        batch_size=32
    )

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    images = next(test_data_generator)
    predictions = loaded_model.predict(images)
    label_dict = {0: 'Flower', 1:'Muffin', 2:'Wheel'}
    st.success("The image is predicted to be - " + label_dict[np.argmax(predictions, axis=1)[0]])