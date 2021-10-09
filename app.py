import streamlit as st
import tensorflow as tf

def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    return pred_class


def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image / 255.
    return image


class_names = ['Boron', 'Iron', 'Nitrogen', 'Phosphorus', 'Potassium']
model = tf.keras.models.load_model("HarvestAIModel.hdf5")

st.set_page_config(page_title="Xtinguish")

#### Main Body ####

st.title("HarvestAI")
st.write("")
file = st.file_uploader(label="Upload an image",
                        type=["jpg", "jpeg", "png"])

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred = predicting(image, model)
    st.success(f'**Prediction :** {pred} Deficiency')
    







