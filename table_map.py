# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable


import numpy as np
import xml.etree.ElementTree as ET

import torch

import util.misc as utils

from models import build_model
from datasets.tables import make_Table_transforms

import matplotlib.pyplot as plt
import time

from PIL import Image


def get_xml_and_image_paths(xml_files, image_files):
    xml_paths = []
    image_paths = []

    for filename in os.listdir(xml_files):

        if filename.endswith('.xml'):
            xml_file_path = os.path.join(xml_files, filename)
            xml_paths.append(xml_file_path)

            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            image_path = root.find('path').text
            temp_img_path = os.path.join(image_files,image_path)
            image_paths.append(temp_img_path)

    return xml_paths, image_paths


def calculate_iou(bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    intersection_area = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
    union_area = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    precision = []
    recall = []

    for pred_box in pred_boxes:
        max_iou = 0
        match_found = False

        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and iou > max_iou:
                max_iou = iou
                match_found = True

        if match_found:
            tp += 1
        else:
            fp += 1

        fn = len(gt_boxes) - tp

        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    return np.array(precision), np.array(recall)


def calculate_ap(precision, recall):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices])
    return ap




def calculate_map(xml_files, image_paths):
    num_files = len(xml_files)
    ap_sum50 = 0
    ap_sum75 = 0
    ap_sum90 = 0

    for i, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_path = image_paths[i]

        pred_boxes_temp = infer(image_paths[i], model, postprocessors, device, args.output_dir)

        gt_boxes = []
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            gt_boxes.append((xmin, ymin, xmax, ymax))

        pred_boxes = [tuple(box) for box in pred_boxes_temp]

       
        precision1, recall1 = calculate_precision_recall(gt_boxes, pred_boxes, 0.50)
        precision2, recall2 = calculate_precision_recall(gt_boxes, pred_boxes, 0.75)
        precision3, recall3 = calculate_precision_recall(gt_boxes, pred_boxes, 0.90)

        ap50 = calculate_ap(precision1, recall1)
        ap75 = calculate_ap(precision2, recall2)
        ap90 = calculate_ap(precision3, recall3)

        ap_sum50 += ap50
        ap_sum75 += ap75
        ap_sum90 += ap90

    mAP50 = ap_sum50 / num_files
    mAP75 = ap_sum75 / num_files
    mAP90 = ap_sum90 / num_files

    return mAP50, mAP75, mAP90


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='face')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    parser.add_argument('--xml_files')
    parser.add_argument('--image_files')

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    for img_sample in images_path:
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
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
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
            continue
        
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

        return bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    xml_files = args.xml_files
    image_files = args.image_files
    
    xml_paths, image_paths = get_xml_and_image_paths(xml_files, image_files)

    mAP50, mAP75, mAP90 = calculate_map(xml_paths, image_paths)

    print(f"Average Precision (AP) - 0.50 iou  :", AP50)
    print(f"Average Precision (AP) - 0.75 iou  :", AP75)
    print(f"Average Precision (AP) - 0.90 iou  :", AP90)


    
