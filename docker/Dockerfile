FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04

ARG SERVING_PORT=8080
ENV SERVING_PORT=$SERVING_PORT

WORKDIR /app

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
    apt install -y python3-pip && \
    apt install -y libgl1-mesa-glx libglib2.0-0

# Upgrade pip and install the copied in requirements.
RUN pip install --no-cache-dir --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# These need to be pre-downloaded to be used
RUN mkdir assets
RUN mkdir assets/pretrained_models
ADD assets/pretrained_models/79999_iter.pth assets/pretrained_models/.
ADD assets/pretrained_models/GFPGANv1.4.pth assets/pretrained_models/.
ADD assets/pretrained_models/RealESRGAN_x2.pth assets/pretrained_models/.
ADD assets/pretrained_models/RealESRGAN_x4.pth assets/pretrained_models/.
ADD assets/pretrained_models/RealESRGAN_x8.pth assets/pretrained_models/.
ADD assets/pretrained_models/codeformer.onnx assets/pretrained_models/.
ADD assets/pretrained_models/inswapper_128.onnx assets/pretrained_models/.
ADD assets/pretrained_models/open-nsfw.onnx assets/pretrained_models/.
RUN mkdir gfpgan
RUN mkdir gfpgan/weights
ADD gfpgan/weights/detection_Resnet50_Final.pth gfpgan/weights/.
ADD gfpgan/weights/parsing_parsenet.pth gfpgan/weights/.

# Copy in the files necessary to fetch, run and serve the model.
ADD model.py .
ADD server.py .
ADD my_faceswap.py .
ADD face_analyser.py .
ADD face_enhancer.py .
ADD face_swapper.py .
ADD utils.py .

RUN mkdir face_parsing
ADD face_parsing/__init__.py face_parsing/.
ADD face_parsing/model.py face_parsing/.
ADD face_parsing/parse_mask.py face_parsing/.
ADD face_parsing/resnet.py face_parsing/.
ADD face_parsing/swap.py face_parsing/.

RUN mkdir nsfw_checker
ADD nsfw_checker/__init__.py nsfw_checker/.
ADD nsfw_checker/opennsfw.py nsfw_checker/.

RUN mkdir upscaler
ADD upscaler/__init__.py upscaler/.
ADD upscaler/codeformer.py upscaler/.

RUN mkdir upscaler/RealESRGAN
ADD upscaler/RealESRGAN/__init__.py upscaler/RealESRGAN/.
ADD upscaler/RealESRGAN/arch_utils.py upscaler/RealESRGAN/.
ADD upscaler/RealESRGAN/model.py upscaler/RealESRGAN/.
ADD upscaler/RealESRGAN/rrdbnet_arch.py upscaler/RealESRGAN/.
ADD upscaler/RealESRGAN/utils.py upscaler/RealESRGAN/.

# Test data
ADD my_test_image.jpg .

# Fetch the model and cache it locally.
# RUN python3 model.py --fetch

# Expose the serving port.
EXPOSE $SERVING_PORT

# Run the server to handle inference requests.
CMD python3 -u server.py