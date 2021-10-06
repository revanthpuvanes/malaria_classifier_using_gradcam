import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import Model
# from IPython.display import Image, display



def main():
    selected_box = st.sidebar.selectbox(
    'Pick Something Fun',
    ('Welcome','Malaria Detection')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Malaria Detection':
        mal_det()


def welcome():
    
    st.title('MALARIA DETECTION WITH ML')
    st.write("Go to the left sidebar to explore")

def rev(path):

    model_path = path 
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Error Loading Model...")

    return model

def preprocess_image(img_path):  #, target_size=(75, 75)):

    img = img_path.resize((75,75))
    # img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255

    return img

def grad_cam(model, img,
             layer_name="rr", label_name=None,
             category_id=None):
  
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id == None:
            category_id = np.argmax(predictions[0])
        if label_name:
            print(label_name[category_id])
        output = predictions[:, category_id]
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap),category_id

def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):

    # img = cv2.imread(img_path)
    img = image.img_to_array(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)  
    # display(imgwithheat)

    return imgwithheat


def detect(img_path):
    # model = model_from_json(open("./models/cnn_architecture_1.json", "r").read())
    # model.load_weights("./models/malaria_detection_1.h5")
    model = rev('./models/malaria_detection_2.h5')
    classes = ('Defected','Normal')
    # img_path = np.array(image.convert('RGB'))
    # img = cv2.resize(img, (75, 75))
    # img = img_to_array(img)
    # img = np.expand_dims(img, axis = 0)
    # img /= 255
    img = preprocess_image(img_path)
    heatmap,label = grad_cam(model, img,
                   label_name = ['Defected', 'Normal'],
                   #category_id = 0,
                   )
    result = classes[label]
    image = show_imgwithheat(img_path, heatmap)
    return image,result


def mal_det():
    st.title("Malaria Detection")
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp', 'jfif'])
    if image_file is not None:
        
    	img_path = Image.open(image_file)

    	if st.button("Process"):
        
    		result_img,result = detect(img_path)
    		st.image(result_img, use_column_width = 'auto')
    		st.success(result)

    if st.button('See Original Image'):
        if image_file is not None:
            original = Image.open(image_file)
            st.image(original, use_column_width='auto')
        else:
            st.write("Please upload any image using browse files ")


if __name__ == "__main__":
    main()