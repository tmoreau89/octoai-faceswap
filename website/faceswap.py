import streamlit as st
from octoai.client import Client
from io import BytesIO
from base64 import b64encode, b64decode
import requests
from PIL import Image, ExifTags
import os
import time

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

def query_faceswap(payload):
     # Send to faceswap endpoint
    headers = {
        "Content-type": "application/json",
        "X-OctoAI-Async": "1",
        "Authorization": f"Bearer {OCTOAI_TOKEN}",
    }
    url = f"{FACESWAP_ENDPOINT_URL}/generate"
    # Send image
    response = requests.post(url=url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def faceswap(my_upload, meta_prompt):
    # UI columps
    colI, colO = st.columns(2)

    # OctoAI client
    oai_client = Client(OCTOAI_TOKEN)

    # Rotate image and perform some rescaling
    input_img = Image.open(my_upload)
    input_img = rotate_image(input_img)
    input_img = rescale_image(input_img)
    colI.write("Input image")
    colI.image(input_img)
    progress_text = "Face swapping in action..."
    percent_complete = 0
    progress_bar = colO.progress(percent_complete, text=progress_text)

    # Query endpoint async
    future = oai_client.infer_async(
        f"{FACESWAP_ENDPOINT_URL}/predict",
        {
            "image": read_image(input_img)
        }
    )
    # Poll on completion
    time_step = 0.2
    while not oai_client.is_future_ready(future):
        time.sleep(time_step)
        percent_complete = min(99, percent_complete+1)
        if percent_complete == 99:
            progress_text = "Face swapping is taking longer than usual, hang tight!"
        progress_bar.progress(percent_complete, text=progress_text)
    # Process results
    results = oai_client.get_future_result(future)
    progress_bar.empty()
    colO.write("Face swapped images :star2:")
    faceswapped_image = Image.open(BytesIO(b64decode(results["image"])))
    colO.image(faceswapped_image)


st.set_page_config(layout="wide", page_title="Face Swapper")

st.write("## :tada: Face Swapper")
st.write("\n\n")
st.write("### :camera: Magically face swap photos with AI!")

st.sidebar.image("octoml-octo-ai-logo-color.png")
my_upload = st.sidebar.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if st.button('OctoShop!'):
        faceswap(my_upload)
