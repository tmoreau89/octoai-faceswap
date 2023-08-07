"""Model wrapper for serving face swapper from https://github.com/harisreedhar/Swap-Mukham/tree/main."""

import argparse
import typing
import os
import cv2
import torch
import insightface
import numpy as np
from tqdm import tqdm
import random
import tempfile
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode

from nsfw_checker import NSFWChecker
from face_swapper import Inswapper, paste_to_whole
from face_analyser import get_analysed_data
from face_parsing import init_parsing_model, get_parsed_mask, mask_regions_to_list
from face_enhancer import load_face_enhancer_model, cv2_interpolations
from utils import split_list_by_lengths


# GPU parameters
PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
BATCH_SIZE = 32
DEVICE = "cuda"

# FaceSwap params
FACE_ENHANCER_NAME = "GFPGAN"
ENABLE_FACE_PARSER = False


def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

def random_derangement(n):
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)


class Model:
    """Wrapper for CLIP interrogator model."""

    def __init__(self):
        """Initialize the model."""
        print("Loading NSFW detector model...")
        self._nsfw_detector = NSFWChecker(
            model_path="./assets/pretrained_models/open-nsfw.onnx",
            providers=PROVIDER
        )
        print("Loading face analyzer model...")
        self._face_analyzer = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=PROVIDER
        )
        self._face_analyzer.prepare(
            ctx_id=0, det_size=(640, 640), det_thresh= 0.6
        )
        print("Loading face swapper model...")
        self._face_swapper = Inswapper(
            model_file="./assets/pretrained_models/inswapper_128.onnx",
            batch_size=BATCH_SIZE,
            providers=PROVIDER
        )
        if FACE_ENHANCER_NAME != "NONE":
            print("Loading face enhancer model...")
            if FACE_ENHANCER_NAME not in cv2_interpolations:
                print(f"### \n ⌛ Loading {FACE_ENHANCER_NAME} model...")
            self._face_enhancer = load_face_enhancer_model(
                name=FACE_ENHANCER_NAME,
                device=DEVICE
            )
        if ENABLE_FACE_PARSER:
            print("Loading face parser model...")
            self._face_parser = init_parsing_model(
                "./assets/pretrained_models/79999_iter.pth",
                device=DEVICE
            )

    def predict(self, inputs: typing.Dict[str, typing.Any]) -> typing.Dict[str, str]:
        """Return interrogation for the given image.

        :param inputs: dict of inputs containing model inputs
               with the following keys:

        - "image" (mandatory): A base64-encoded image.

        :return: a dict containing these keys:

        - "image": A base64-encoded image.
        """
        mask_includes = [
            "Skin",
            "R-Eyebrow",
            "L-Eyebrow",
            "L-Eye",
            "R-Eye",
            "Nose",
            "Mouth",
            "L-Lip",
            "U-Lip"
        ]
        mask_soft_iterations = 10
        blur_amount = 0.1
        erode_amount = 0.15
        face_scale = 1
        enable_laplacian_blend = True
        crop_top = 0
        crop_bott = 511
        crop_left = 0
        crop_right = 511

        # Process inputs
        image = inputs.get("image", None)
        image_bytes = BytesIO(b64decode(image))
        image_pil = Image.open(image_bytes).convert('RGB')

        # Target path
        ntf_target = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_path = ntf_target.name
        # Source path (same as target)
        source_path = image_path
        image_pil.save(source_path)
        # Output path
        ntf_output = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        output_file = ntf_output.name

        # Write the output file for now
        image_pil.save(output_file)
        image_sequence = [output_file]

        # Prepare inputs
        includes = mask_regions_to_list(mask_includes)
        if crop_top > crop_bott:
            crop_top, crop_bott = crop_bott, crop_top
        if crop_left > crop_right:
            crop_left, crop_right = crop_right, crop_left
        crop_mask = (crop_top, 511-crop_bott, crop_left, 511-crop_right)


        ## ------------------------------ CONTENT CHECK ------------------------------
        print("### \n ⌛ Checking contents...")
        nsfw = self._nsfw_detector .is_nsfw(image_sequence)
        if nsfw:
            message = "NSFW Content detected !!!"
            print(f"### \n 🔞 {message}")
            assert not nsfw, message
        torch.cuda.empty_cache()

        ## ------------------------------ ANALYSE FACE ------------------------------
        print("### \n ⌛ Analyzing face data...")
        age = 25
        source_data = source_path, age
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            self._face_analyzer,
            image_sequence,
            source_data,
            swap_condition="All Face",
            detect_condition="best detection",
            scale=face_scale
        )
        perm = random_derangement(len(analysed_targets))
        analysed_sources = []
        for perm_idx in perm:
            analysed_sources.append(analysed_targets[perm_idx])
        for idx, targ in enumerate(analysed_sources):
            print("SRC FACE #{}".format(idx))
            print("\t bbox = {}".format(targ['bbox']))
            print("\t gender = {}".format(targ['gender']))
            print("\t age = {}".format(targ['age']))

        ## ------------------------------ SWAP FUNC ------------------------------
        print("### \n ⌛ Generating faces...")
        preds = []
        matrs = []
        count = 0
        # global PREVIEW
        for batch_pred, batch_matr in self._face_swapper.batch_forward(whole_frame_list, analysed_targets, analysed_sources):
            preds.extend(batch_pred)
            matrs.extend(batch_matr)
            torch.cuda.empty_cache()
            count += 1

        ## ------------------------------ FACE ENHANCEMENT ------------------------------
        generated_len = len(preds)
        if FACE_ENHANCER_NAME != "NONE":
            print(f"### \n ⌛ Upscaling faces with {FACE_ENHANCER_NAME}...")
            for idx, pred in tqdm(enumerate(preds), total=generated_len, desc=f"Upscaling with {FACE_ENHANCER_NAME}"):
                enhancer_model, enhancer_model_runner = self._face_enhancer
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512,512))
        torch.cuda.empty_cache()

        ## ------------------------------ FACE PARSING ------------------------------
        if ENABLE_FACE_PARSER:
            print("### \n ⌛ Face-parsing mask...")
            masks = []
            count = 0
            for batch_mask in get_parsed_mask(self._face_parser, preds, classes=includes, device=DEVICE, batch_size=32, softness=int(mask_soft_iterations)):
                masks.append(batch_mask)
                torch.cuda.empty_cache()
                count += 1
            masks = np.concatenate(masks, axis=0) if len(masks) >= 1 else masks
        else:
            masks = [None] * generated_len

        ## ------------------------------ SPLIT LIST ------------------------------
        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs
        split_masks = split_list_by_lengths(masks, num_faces_per_frame)
        del masks

        ## ------------------------------ PASTE-BACK ------------------------------
        print("### \n ⌛ Pasting back...")
        for frame_idx, frame_img in enumerate(image_sequence):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            blend_method = 'laplacian' if enable_laplacian_blend else 'linear'
            for p, m, mask in zip(split_preds[frame_idx], split_matrs[frame_idx], split_masks[frame_idx]):
                p = cv2.resize(p, (512,512))
                mask = cv2.resize(mask, (512,512)) if mask is not None else None
                m /= 0.25
                whole_img = paste_to_whole(p, whole_img, m, mask=mask, crop_mask=crop_mask, blend_method=blend_method, blur_amount=blur_amount, erode_amount=erode_amount)
            cv2.imwrite(whole_img_path, whole_img)

        output_image = Image.open(output_file)
        response = {"image": read_image(output_image)}

        # Clean up
        ntf_target.close()
        ntf_output.close()
        os.unlink(ntf_target.name)
        os.unlink(ntf_output.name)

        return response

    @classmethod
    def fetch(cls) -> None:
        """Pre-fetches the model for implicit caching by Transfomers."""
        # Running the constructor is enough to fetch this model.
        cls()

def main():

    """Entry point for interacting with this model via CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    if args.fetch:
        Model.fetch()

if __name__ == "__main__":
    main()


