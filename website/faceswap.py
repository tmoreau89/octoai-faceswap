import streamlit as st
from io import BytesIO
from base64 import b64encode, b64decode
import requests
from PIL import Image, ExifTags
import os

FACESWAP_ENDPOINT_URL = os.environ["FACESWAP_ENDPOINT_URL"]
OCTOAI_TOKEN = os.environ["OCTOAI_TOKEN"]

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

def rotate_image(image):
    try:
        # Rotate based on Exif Data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image
    except:
        return image

def rescale_image(image):
    w, h = image.size
    if w == h:
        width = 1024
        height = 1024
    elif w > h:
        width = 1024
        height = 1024 * h // w
    else:
        width = 1024 * w // h
        height = 1024
    image = image.resize((width, height))
    return image

def faceswap(my_upload):
    # UI columps
    colI, colO = st.columns(2)

    # Rotate image and perform some rescaling
    input_img = Image.open(my_upload)
    input_img = rotate_image(input_img)
    input_img = rescale_image(input_img)
    colI.write("Input image")
    colI.image(input_img)

    # Query endpoint async
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {OCTOAI_TOKEN}",
    }
    url = f"{FACESWAP_ENDPOINT_URL}/predict"
    response = requests.post(url=url, json={"image": read_image(input_img)}, headers=headers)
    # Process results
    colO.empty()
    colO.write("Face swapped images :star2:")
    faceswapped_image = Image.open(BytesIO(b64decode(response.json()["completion"]["image"])))
    colO.image(faceswapped_image)


st.set_page_config(layout="wide", page_title="AI Face Swapper")

st.write("## :tada: AI Face Swapper, powered by OctoAI")
st.write("\n\n")
st.write("### :camera: Magically swap faces with AI!")

# st.sidebar.image("octoml-octo-ai-logo-color.png")
my_upload = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if st.button('Face Swap!'):
        faceswap(my_upload)
