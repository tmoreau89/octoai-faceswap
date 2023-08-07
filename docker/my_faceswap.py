import os
import cv2
import torch
import insightface
import numpy as np
from tqdm import tqdm
import concurrent.futures
import random

from nsfw_checker import NSFWChecker
from face_swapper import Inswapper, paste_to_whole
from face_analyser import get_analysed_data
from face_parsing import init_parsing_model, get_parsed_mask, mask_regions_to_list
from face_enhancer import load_face_enhancer_model, cv2_interpolations
from utils import split_list_by_lengths

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None
NSFW_DETECTOR = None

# File Paths
target_im_path = "my_test_image.jpg"
source_im_path = target_im_path
output_dir = "."
output_name = "output"

PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DETECT_SIZE = 640
DETECT_THRESH = 0.6
BATCH_SIZE = 32
DETECT_CONDITION = "best detection"
device = "cuda"

input_type = "Image"
image_path = target_im_path
source_path = source_im_path
output_path = output_dir
output_name = output_name
keep_output_sequence = False
condition = "All Face"
age = 25
distance = 0.6
face_enhancer_name = "GFPGAN"
enable_face_parser = True
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
mask_soft_kernel = 17
mask_soft_iterations = 10
blur_amount = 0.1
erode_amount = 0.15
face_scale = 1
enable_laplacian_blend = True
crop_top = 0
crop_bott = 511
crop_left = 0
crop_right = 511
specifics = [None, None, None, None, None, None, None, None, None, None]

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

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )

def load_face_swapper_model(path="./assets/pretrained_models/inswapper_128.onnx"):
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(model_file=path, batch_size=batch, providers=PROVIDER)

def load_face_parser_model(path="./assets/pretrained_models/79999_iter.pth"):
    global FACE_PARSER
    if FACE_PARSER is None:
        FACE_PARSER = init_parsing_model(path, device=device)

def load_nsfw_detector_model(path="./assets/pretrained_models/open-nsfw.onnx"):
    global NSFW_DETECTOR
    if NSFW_DETECTOR is None:
        NSFW_DETECTOR = NSFWChecker(model_path=path, providers=PROVIDER)

if __name__ == "__main__":
    # Load models
    # print("Loading NSFW detector model...")
    # load_nsfw_detector_model()
    print("Loading face analyzer model...")
    load_face_analyser_model()
    print("Loading face swapper model...")
    load_face_swapper_model()

    if face_enhancer_name != "NONE":
        if face_enhancer_name not in cv2_interpolations:
            print(f"### \n ⌛ Loading {face_enhancer_name} model...")
        FACE_ENHANCER = load_face_enhancer_model(name=face_enhancer_name, device=device)
    else:
        FACE_ENHANCER = None

    if enable_face_parser:
        print("Loading face parser model...")
        load_face_parser_model()

    # Prepare inputs
    includes = mask_regions_to_list(mask_includes)
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]
    if crop_top > crop_bott:
        crop_top, crop_bott = crop_bott, crop_top
    if crop_left > crop_right:
        crop_left, crop_right = crop_right, crop_left
    crop_mask = (crop_top, 511-crop_bott, crop_left, 511-crop_right)

    def swap_process(image_sequence):
        ## ------------------------------ CONTENT CHECK ------------------------------

        # print("### \n ⌛ Checking contents...")
        # nsfw = NSFW_DETECTOR.is_nsfw(image_sequence)
        # if nsfw:
        #     message = "NSFW Content detected !!!"
        #     print(f"### \n 🔞 {message}")
        #     assert not nsfw, message
        # torch.cuda.empty_cache()

        ## ------------------------------ ANALYSE FACE ------------------------------

        print("### \n ⌛ Analysing face data...")
        if condition != "Specific Face":
            source_data = source_path, age
        else:
            source_data = ((sources, specifics), distance)
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=condition,
            detect_condition=DETECT_CONDITION,
            scale=face_scale
        )
        # print("analysed_targets: {}".format(analysed_targets))
        for idx, targ in enumerate(analysed_targets):
            print("TARG FACE #{}".format(idx))
            print("\t bbox = {}".format(targ['bbox']))
            print("\t gender = {}".format(targ['gender']))
            print("\t age = {}".format(targ['age']))
        # print("analysed_sources: {}".format(analysed_sources))
        for idx, targ in enumerate(analysed_sources):
            print("SRC FACE #{}".format(idx))
            print("\t bbox = {}".format(targ['bbox']))
            print("\t gender = {}".format(targ['gender']))
            print("\t age = {}".format(targ['age']))
        print("whole_frame_list: {}".format(whole_frame_list))
        print("num_faces_per_frame: {}".format(num_faces_per_frame))

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
        for batch_pred, batch_matr in FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources):
            preds.extend(batch_pred)
            matrs.extend(batch_matr)
            torch.cuda.empty_cache()
            count += 1

        ## ------------------------------ FACE ENHANCEMENT ------------------------------

        generated_len = len(preds)
        if face_enhancer_name != "NONE":
            print(f"### \n ⌛ Upscaling faces with {face_enhancer_name}...")
            for idx, pred in tqdm(enumerate(preds), total=generated_len, desc=f"Upscaling with {face_enhancer_name}"):
                enhancer_model, enhancer_model_runner = FACE_ENHANCER
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512,512))
        torch.cuda.empty_cache()

        ## ------------------------------ FACE PARSING ------------------------------

        if enable_face_parser:
            print("### \n ⌛ Face-parsing mask...")
            masks = []
            count = 0
            for batch_mask in get_parsed_mask(FACE_PARSER, preds, classes=includes, device=device, batch_size=BATCH_SIZE, softness=int(mask_soft_iterations)):
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

    ## ------------------------------ IMAGE ------------------------------
    target = cv2.imread(image_path)
    output_file = os.path.join(output_path, output_name + ".png")
    cv2.imwrite(output_file, target)
    swap_process([output_file])

    # swap_inputs = [
    #     input_type,               input_type
    #     image_input,              image_path
    #     video_input,              video_path
    #     direc_input,              directory_path
    #     source_image_input,       source_path
    #     output_directory,         output_path
    #     output_name,              output_name
    #     keep_output_sequence,     keep_output_sequence
    #     swap_option,              condition
    #     age,                      age
    #     distance_slider,          distance
    #     face_enhancer_name,       face_enhancer_name
    #     enable_face_parser_mask,  enable_face_parser
    #     mask_include,             mask_includes
    #     mask_soft_kernel,         mask_soft_kernel
    #     mask_soft_iterations,     mask_soft_iterations
    #     blur_amount,              blur_amount
    #     erode_amount,             erode_amount
    #     face_scale,               face_scale
    #     enable_laplacian_blend,   enable_laplacian_blend
    #     crop_top,                 crop_top
    #     crop_bott,                crop_bott
    #     crop_left,                crop_left
    #     crop_right,               crop_right
    #     *src_specific_inputs,     *specifics
    # ]
