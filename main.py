from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os 
from transformers import pipeline
import streamlit as st
from PIL import Image
os.environ['HF_HOME']='D:/huggingface_cache'

captioner=pipeline(
    "image-to-text",
    model='nlpconnect/vit-gpt2-image-captioning',
    framework='pt'
)
# image=Image.open('6.jpg')

# result=captioner(image)
# print(result[0]['generated_text'])

st.title("Image Caption Generator")
image_input=st.file_uploader("Choose the Image:",type=['jpg','png','jpeg'])

if image_input is not None:
    open_image=Image.open(image_input)
    st.image(open_image,caption='Uploaded image',use_container_width=True)
    with st.spinner("Generating caption..."):
        result=captioner(open_image)

    st.subheader("Generated Caption:")
    st.success(result[0]['generated_text'])
else:
    st.info("Please upload an image to generate a caption.")

