from __future__ import print_function
import os
import glob
import dlib
import torch
import torch.nn as nn
from enum import Enum
from skimage import io
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from face_alignment.models import FAN, ResNetDepth
from face_alignment.utils import *
from face_alignment.api import (LandmarksType, NetworkSize)
from face_alignment import FaceAlignment as FaceAlignmentBase


def get_preds_fromhm(hm, center=None, scale=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig, max


class Adapter(FaceAlignmentBase):
    def __init__(self):
        try:
            super().__init__(
                LandmarksType._2D,
                enable_cuda=True,
                flip_input=False)
        except:
            super().__init__(
                LandmarksType._2D,
                device='cuda',
                flip_input=False)

    def get_landmarks(self, image, d=None):
        with torch.no_grad():
            if d is not None:
                center = torch.FloatTensor(
                    [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                     (d.bottom() - d.top()) / 2.0])
                center[1] = center[1] - (d.bottom() - d.top()) * 0.12
                scale = (d.right() - d.left() +
                         d.bottom() - d.top()) / 195.0
            else:
                right = image.shape[0]
                left = 0
                bottom = image.shape[1]
                top = 0
                center = torch.FloatTensor(
                    [right - (right - left) / 2.0, bottom -
                     (bottom - top) / 2.0])
                center[1] = center[1] - (bottom - top) * 0.12
                scale = (right - left +
                         bottom - top) / 195.0

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float().div(255.0).unsqueeze_(0)

            inp = inp.cuda()

            if hasattr(self, 'face_alignemnt_net'):
                out = self.face_alignemnt_net(inp)
            else:
                out = self.face_alignment_net(inp)

            out = out[-1].detach().cpu()

            pts, pts_img, confidence = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            return pts_img.numpy(), confidence.numpy()[0]

