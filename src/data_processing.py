import copy
import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import torchvision


class ModelData(object):
    """Class for generating model information(pre-processing arguments)."""

    def __init__(self,
                 input_shape=(1, 3, 640, 640),
                 class_name='datasets/annotations/class.names',
                 conf_thres=0.35,
                 iou_thres=0.65,
                 ):
        """Initialize with all values that will be used in pre or post processing.

        Keyword arguments:
        input_shape -- The tensor shape of the batch to prepare, either in NCHW or NHWC format
        class_name -- The path of label list
        conf_thres -- Object confidence threshold
        iou_thres -- IOU threshold for NMS
        """
        self.preprocessor_args = {
            "input_shape": input_shape,
            "scale": 255.0,
            "mean": [0., 0., 0.],
            "std": [1.0, 1.0, 1.0],
            "letterbox": True
        }
        self.labels = read_class_names(class_name)
        self.postprocessor_args = {
            "masks": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            "anchors": [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                        [72, 146], [142, 110], [192, 243], [459, 401]],
            "strides": [16, 32, 64],
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "classes": self.labels,
            "multi_label": True,
            "max_det": 1000
        }

    def get_preprocessor_args(self):
        """get pre processing arguments"""
        return self.preprocessor_args

    def get_postprocessor_args(self):
        """get post processing arguments"""
        return self.postprocessor_args


class Postprocessor(object):
    """Class for post-processing the three outputs tensors from YOLOv7."""

    def __init__(self,
                 input_size,
                 ):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of YOLOv7.

        Keyword arguments:
        input_size -- The input size of image
        """

        self.width = input_size[3]
        self.height = input_size[2]

    def draw_bboxes(
            self,
            image_path,
            bboxes,
            confidences,
            categories,
            all_categories,
            bbox_color='blue',
    ):
        """Draw the bounding boxes on the original input image and return it.

        Keyword arguments:
        image_path -- the path of image
        bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
        categories -- NumPy array containing the corresponding category for each object,
        with shape (N,)
        confidences -- NumPy array containing the corresponding confidence for each object,
        with shape (N,)
        all_categories -- a list of all categories in the correct ordered (required for looking up
        the category name)
        bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
        """
        np.random.seed(1)
        colors = [[np.random.randint(0, 255)
                   for _ in range(3)]
                  for _ in range(len(all_categories))]

        image_raw = Image.open(image_path)
        image_mode = image_raw.mode
        draw = ImageDraw.Draw(image_raw)
        text_size = 20
        font = ImageFont.load_default()

        if bboxes is not None:
            for box, score, category in zip(bboxes, confidences, categories):
                left, top, right, bottom = box
                bbox_color = 255 if image_mode == 'L' else tuple(
                    colors[int(category)])
                draw.rectangle(((left - right / 2, top - bottom / 2), (left + right / 2, top + bottom / 2)),
                               outline=bbox_color)
                draw.text((left, top - 20),
                          "{0} {1:.2f}".format(all_categories[category], score),
                          fill=bbox_color,
                          font=font)
        else:
            pass

        return image_raw

    def parse_detection_output(
            self,
            detections,
            resolution_raw,
            imgsz,
            letterbox,
    ):
        """Parse [boxes, conf, classes] detection outputs to [boxes], [conf], [classes] format"""
        boxes = []
        classes = []
        scores = []
        detections = detections.cpu().detach().numpy()
        for i, det in enumerate(detections):
            xyxy = det[:4]
            conf = det[4]
            cls = det[5]
            if letterbox:
                box = self.letterbox_reverse(
                    xyxy,
                    resolution_raw,
                    imgsz,
                )
            else:
                box = self.rescale_bbox(
                    xyxy,
                    resolution_raw,
                    imgsz,
                )
            boxes.append(box)
            classes.append(int(cls))
            scores.append(float(conf))

        return boxes, classes, scores

    def rescale_bbox(self, boxes, resolution_raw, imgsz):
        """rescale bounding box using original image's size and model's input size"""
        origin_width, origin_height = resolution_raw
        width, height = imgsz

        if len(boxes) == 0:
            return boxes

        ratio_x, ratio_y = width / origin_width, height / origin_height

        boxes[0] = np.clip((boxes[0]) / ratio_x, 0, origin_width)
        boxes[2] = np.clip((boxes[2]) / ratio_x, 0, origin_width)
        boxes[1] = np.clip((boxes[1]) / ratio_y, 0, origin_height)
        boxes[3] = np.clip((boxes[3]) / ratio_y, 0, origin_height)

        return boxes

    def letterbox_reverse(
            self,
            boxes,
            resolution_raw,
            imgsz,
    ):
        """reverse letterbox bounding box cooordination using original image's size and model's input size"""
        origin_width, origin_height = resolution_raw
        width, height = imgsz
        if len(boxes) == 0:
            return boxes

        ratio = min(width / origin_width, height / origin_height)
        resize_w, resize_h = int(origin_width * ratio), int(origin_height *
                                                            ratio)
        x_pad, y_pad = (width - resize_w) // 2, (height - resize_h) // 2
        ratio_x, ratio_y = width / origin_width, height / origin_height

        boxes[0] = np.clip((boxes[0] - x_pad) / ratio, 0, origin_width)
        boxes[2] = np.clip((boxes[2] - x_pad) / ratio, 0, origin_width)
        boxes[1] = np.clip((boxes[1] - y_pad) / ratio, 0, origin_height)
        boxes[3] = np.clip((boxes[3] - y_pad) / ratio, 0, origin_height)

        return boxes


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def non_max_suppression(
        prediction,
        masks=None,
        anchors=None,
        strides=None,
        conf_thres=0.25,
        iou_thres=0.65,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
):
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, outputs = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference outputs

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], 1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    # multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label = multi_label and (nc > 1)  # multiple labels per box (adds 0.5ms/img)

    prediction = np.swapaxes(prediction, -1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction = torch.from_numpy(prediction)
    if not rotated:
        if in_place:
            prediction[..., :4] = _xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((_xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = _xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        classes = None
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        scores = x[:, 4]  # scores

        boxes = _xywh2xyxy(x[..., :4])  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return output


# def non_max_suppression(prediction,
#                         masks=None,
#                         anchors=None,
#                         strides=None,
#                         conf_thres=0.35,
#                         iou_thres=0.65,
#                         classes=None,
#                         max_det=300,
#                         agnostic=False,
#                         multi_label=True,
#                         labels=()):
#     """Runs Non-Maximum Suppression (NMS) on inference results
#
#     Returns:
#          list of detections, on (n,6) tensor per image [xyxy, conf, cls]
#     """
#     print(prediction.shape)
#     nc = prediction.shape[2] - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates
#
#     # Settings
#     min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     time_limit = 10.0  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label = multi_label and (nc > 1)  # multiple labels per box (adds 0.5ms/img)  <-  multi_label &= nc > 1
#     merge = False  # use merge-NMS
#
#     t = time.time()
#     prediction = torch.from_numpy(prediction)
#     outputs = [torch.zeros(
#         (0, 6), device=prediction.device)] * prediction.shape[0]
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence
#
#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             l = labels[xi]
#             v = torch.zeros((len(l), nc + 5), device=x.device)
#             v[:, :4] = l[:, 1:5]  # box
#             v[:, 4] = 1.0  # conf
#             v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
#             x = torch.cat((x, v), 0)
#
#         # If none remain process next image
#         if not x.shape[0]:
#             continue
#
#         # Compute conf
#         if nc == 1:
#             x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
#             # so there is no need to multiplicate.
#         else:
#             x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
#
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = _xywh2xyxy(x[:, :4])
#
#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#
#         # Filter by class
#         classes = None  ####
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]
#
#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             x = x[x[:,
#                     4].argsort(descending=True)[:max_nms]]  # sort by confidence
#
#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n <
#                       3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = torchvision.ops.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
#                 1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy
#
#         outputs[xi] = x[i]
#         if (time.time() - t) > time_limit:
#             print(f'WARNING: NMS time limit {time_limit}s exceeded')
#             break  # time limit exceeded
#     return outputs


def _xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
                 ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / (
                 (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
         ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
                 ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
                 / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
                 + eps
         ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd
