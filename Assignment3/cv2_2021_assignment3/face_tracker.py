# Copyright (c) 3DUniversum BV. All rights reserved.

# THIS SOFTWARE IS PROVIDED BY 3DUNIVERSUM B.V. "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import math

import cv2
import dlib
import openface

import numpy as np

from adapters.DLIB import Adapter


class FaceNotFoundError(Exception):
    def __init__(self):
        super(FaceNotFoundError, self).__init__("face not found")


class CNNDetector:
    def __init__(self, weight_path="../models/mmod_human_face_detector.dat"):
        self.detect = dlib.cnn_face_detection_model_v1(weight_path)

    def __call__(self, bgr, scale_ratio=1):
        dets = self.detect(bgr[:,:,::-1], scale_ratio)
        if len(dets) == 0:
            raise FaceNotFoundError()

        return max(dets, key=lambda d: d.rect.area()).rect


class CascadeDetector:
    def __init__(self, dlib_weight_path):
        self.align = openface.AlignDlib(dlib_weight_path)

    def __call__(self, bgr, scale_ratio=1):
        gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        d = self.align.getLargestFaceBoundingBox(gray_image)
        if d is None:
            raise FaceNotFoundError()

        return d

class FaceTracker:
    def __init__(self, dlib_weight_path, use_unconfident_tracker=False,
                 threshold=15, border_fraction=0.1, landmark_detector=None,
                 use_model_cnn=True):
        """ Class for tracking the face. Returns a bounding box with border expansion w.r.t.
        facial landmarks bounding box.

        DLIB weights can be downloaded from http://dlib.net/. Make sure dlib is compiled with
        CUDA when use_model_cnn=True is used.

        landmark_detector is Dlib by default, but others can be found from adapters/*. Dependent
        on the accuracy of landmark accuracy border_fraction need to be changed to cover the face
        region.

        if use_unconfident_tracker==True, face detector will be executed only once (at the beginning),
        otherwise, it will be executed when confidence is below the threshold.

        Use FaceTracker.track method for tracking.
        """

        if landmark_detector is None:
            self.landmark_detector = Adapter(dlib_weight_path)
        else:
            self.landmark_detector = landmark_detector

        if use_model_cnn:
            self.detect = CNNDetector(dlib_weight_path)
        else:
            self.detect = CascadeDetector(dlib_weight_path)

        self.face_box = None
        self.tracker = dlib.correlation_tracker()
        self.use_unconfident_tracker = use_unconfident_tracker
        self.threshold = threshold
        self.border_fraction = border_fraction
        self.is_tracker_reset = False
        self.confidence = None

    def start_tracking(self, frame, scale_ratio):
        print("Start tracking!")
        self.is_tracker_reset = True
        face_box = self.detect(frame, scale_ratio)
        self.tracker.start_track(frame, face_box)

        return face_box

    def update_box_using_landmarks(self, frame, box):
        lm, confidence = self.landmark_detector.get_landmarks(frame, box)

        box = self.compute_face_box(frame, lm)

        return box, np.asarray(lm, dtype=np.int32), confidence


    def track(self, frame, scale_ratio=2):
        if self.face_box is None:
            self.start_tracking(frame, scale_ratio)
        else:
            tracking_quality = self.tracker.update(frame)
            if (tracking_quality < self.threshold) and not self.use_unconfident_tracker:
                self.start_tracking(frame, scale_ratio)

        box = self.tracker.get_position()
        if math.isnan(box.left()):
            self.start_tracking(frame, scale_ratio)
            box = self.tracker.get_position()
            if math.isnan(box.left()):
                self.is_tracker_reset = True
                self.face_box = None
                raise FaceNotFoundError()

        self.face_box = dlib.rectangle(
            int(box.left()), int(box.top()),
            int(box.right()), int(box.bottom()))

        self.face_box, lm, self.confidence = self.update_box_using_landmarks(frame, self.face_box)

        self.tracker.start_track(frame, self.face_box)
        return self.face_box, lm

    def compute_face_box(self, frame, lm):
        x_min, y_min = np.min(lm, axis=0)
        x_max, y_max = np.max(lm, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        x_min -= width * self.border_fraction
        x_max += width * self.border_fraction
        y_min -= height * self.border_fraction
        y_max += height * self.border_fraction

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        max_width = max(x_max - x_min, y_max - y_min)
        x_min = max(int(x_center - max_width / 2), 0)
        y_min = max(int(y_center - max_width / 2), 0)

        x_max = min(int(x_min + max_width), frame.shape[1])
        y_max = min(int(y_min + max_width), frame.shape[0])
        x_min = x_max - max_width
        y_min = y_max - max_width

        return dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
