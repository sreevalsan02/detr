import math
import os
import cv2
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import util.misc as utils
from models import build_model
from datasets.tables import make_Table_transforms
import matplotlib.pyplot as plt
import time
from PIL import Image
from xml.etree import ElementTree as ET


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def infer(img_sample, model, postprocessors, device, output_path, args):
    model.eval()
    filename = os.path.basename(img_sample)
    orig_image = Image.open(img_sample)
    w, h = orig_image.size
    transform = make_Table_transforms("val")
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    image, targets = transform(orig_image, dummy_target)
    image = image.unsqueeze(0)
    image = image.to(device)

    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1])),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_attn_weights.append(output[1]))
    ]

    start_t = time.perf_counter()
    outputs = model(image)
    end_t = time.perf_counter()

    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > args.thresh

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
    probas = probas[keep].cpu().data.numpy()

    for hook in hooks:
        hook.remove()

    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0].cpu()

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    if len(bboxes_scaled) == 0:
        return []
      
    bounding_boxes = []

    for idx, box in enumerate(bboxes_scaled):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ])
        bbox = bbox.reshape((4, 2))
        bounding_boxes.append(bbox)

    return bounding_boxes


def table_evaluate(img_path, checkpoint):
    args = get_default_args()
    args.resume = checkpoint

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    return infer(img_path, model, postprocessors, device, args.output_dir, args)


def get_default_args():
    class Args:
        def __init__(self):
            self.resume = ''
            self.output_dir = ''
            self.device = 'cuda'
            self.thresh = 0.5

    args = Args()
    return args
