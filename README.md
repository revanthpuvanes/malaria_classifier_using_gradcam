# Malaria Classifier with Hotspot Detection

Have you ever wondered how your trained classifier model actually classifies the right class.

In general you train your model and make predictions out of it.

But here we are going to hotspot the area or pixels where the model sees to predict the right class.

# Demo Video


https://user-images.githubusercontent.com/80465899/148053206-41e9789e-dcdd-4aad-86e8-7c90077b1cc5.mp4

Let's take a look at what we have done.
  - First we train the normal classifier model.
  - Usually we apply softmax in activation layer and predict classes.
  - But here we took the last convolution layer values and name it to later access.
  - And we store the predicted pixels and gradients.
  - Then we multiply gradients with conv values by reduce mean.
  - Finally we apply the heatmap on the predicted pixels.

# Requirements
  - numpy==1.19.5
  - opencv-contrib-python-headless
  - Pillow 
  - Streamlit
  - Tensorflow
  - Keras
  - ipython
