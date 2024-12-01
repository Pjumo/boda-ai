#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

import math
import numpy as np
import cv2


class ImageBatcher:
    """Creates batches of pre-processed images."""

    def __init__(self,
                 input,
                 input_shape,
                 scale,
                 mean,
                 std,
                 letterbox=False,                 
                 max_num_images=None,                 
                 resize_option="BILINEAR",
                 exact_batches=False
                 ):
        """Initialize with all values that used in generating batched images.

        Keyword arguments:
        input: The input directory to read images from.
        input_shape -- The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        scale -- Normalize a tensor image with scale value.
        mean -- Normalize a tensor image with mean value.
        std -- Normalize a tensor image with standard deviation value.
        resize_option -- The option for resizing image e.g. 'BILINEAR' or 'BICUBIC'.
        max_num_images -- The maximum number of images to read from the directory.
        exact_batches -- This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        letterbox -- To keep the proportion of the image, put letterbox to the image.
        """
        # Find images in the given input path
        input = os.path.realpath(input)

        self.images = []
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(
                path)[1].lower() in extensions

        def get_image_paths(base="./"):
            imgs = []
            if os.path.isdir(base):
                for sub in os.listdir(base):
                    imgs += get_image_paths(os.path.join(base, sub))
            elif is_image(base):
                imgs.append(base)
            return sorted(imgs)

        self.images = get_image_paths(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print(f"No valid {'/'.join(extensions)} images found in {input}")
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = np.float32
        self.input_shape = input_shape

        assert len(self.input_shape) == 4
        if self.input_shape[1] == 3:
            self.format = "NCHW"
            self.batch_size, _, self.height, self.width = input_shape
        elif self.input_shape[3] == 3:
            self.format = "NHWC"
            self.batch_size, self.height, self.width, _ = input_shape

        assert self.batch_size > 0
        assert all([self.format, self.width > 0, self.height > 0])
        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (
                self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        # Resize values for preprocessing image
        if resize_option.upper() == "BILINEAR":
            self.resize_option = cv2.INTER_LINEAR
        elif resize_option.upper() == "BICUBIC":
            self.resize_option = cv2.INTER_CUBIC
        else:
            raise ValueError("Wrong resize option. Please check resize option.")

        self.resize_size = (self.width, self.height)

        # Normalization value
        self.scale = scale
        self.mean = mean
        self.std = std

        self.letterbox = letterbox

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.
        This includes cropping, resizing, normalization, data type casting, and transposing.

        Keyword arguments:
        image_path -- The path to the image on disk to load.

        Return:
        A numpy array holding the image sample, ready to be contacatenated into the rest of the batch.
        """

        im0 = cv2.imread(image_path)  # BGR
        new_width, new_height = self.resize_size[0], self.resize_size[1]
        stride = 32

        def make_divisible(x, divisor):
            # Returns nearest x divisible by divisor
            return math.ceil(x / divisor) * divisor

        def check_img_size(imgsz, s=32, floor=0):
            # Verify image size is a multiple of stride s in each dimension
            if isinstance(imgsz, int):  # integer i.e. img_size=640
                new_size = max(make_divisible(imgsz, int(s)), floor)
            else:  # list i.e. img_size=[640, 480]
                imgsz = list(imgsz)  # convert to list if tuple
                new_size = [
                    max(make_divisible(x, int(s)), floor) for x in imgsz
                ]
            if new_size != imgsz:
                print(
                    f'WARNING --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}'
                )
            return new_size

        def letterbox(im,
                      new_shape=(640, 640),
                      color=(114, 114, 114),
                      auto=True,
                      scaleFill=False,
                      scaleup=True,
                      stride=32):
            # Resize and pad image while meeting stride-multiple constraints
            shape = im.shape[:2]  # current shape [height, width]
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better val mAP)
                r = min(r, 1.0)

            # Compute padding
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
                1]  # wh padding
            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            elif scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                new_unpad = (new_shape[1], new_shape[0])
                ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                    0]  # width, height ratios

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                im = cv2.resize(im, new_unpad, interpolation=self.resize_option)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(
                im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                value=color)  # add border
            return im

        if self.letterbox:
            im = letterbox(
                im0, (new_height, new_width), stride=stride, #(H,W)
                auto=False)  # padded resize

        else:
            im = cv2.resize(
                im0, (new_width, new_height), interpolation=self.resize_option)

        im = im / self.scale
        for i in range(0, 3):
            im[:, :, i] = (im[:, :, i] - self.mean[i]) / self.std[i]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return im

    def get_batch(self):
        """Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for raw_img_arr_list, img_arr_list, img_path_list in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        return: A generator yielding three items per iteration:
        a numpy array holding a batch of original images,
        a numpy array holding a batch of pre-processed images,
        and the list of paths to the images loaded within this batch.
        """

        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.input_shape, dtype=self.dtype)
            raw_batch_data = []
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = self.preprocess_image(image)

                image_raw = cv2.imread(image)
                raw_batch_data.append(image_raw)

            self.batch_index += 1
            yield batch_data, [raw_batch_data, batch_images]
