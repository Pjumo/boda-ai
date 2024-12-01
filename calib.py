import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import onnx
import onnxruntime as ort

from sapeon_utils import calibration
from src import ModelData, ImageBatcher
from utils.utils import make_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate ONNX model.")
    parser.add_argument(
        "--model",
        default="./models/yolov8-object.onnx",
        type=str,
        help="Path to the target model in '*.onnx' format.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="A batch size for model's inference.")        
    parser.add_argument(
        "--img_size",
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height and width of input tensor.")        
    parser.add_argument(
        "--calib_data",
        type=str,
        default="./datasets/valid/images/",
        help="Path to calibration dataset.")           
    parser.add_argument(
        "--calib_table",
        type=str,
        default="./calib_tables/yolov8s_high.json",
        help="Path of a calibration table. Use a result of `calib.py`")         
    parser.add_argument(
        "--num_images",
        default=500,
        type=int,
        help="The number of image(s) used in calibration.")
    parser.add_argument(
        "--mode",
        default="medium",
        choices=["high", "medium", "low", "max", "entropy", "percentile"],
        type=str,
        help="Calibration mode.")
    parser.add_argument(
        "--percentile",
        default=99.999,
        type=float,
        help="Percentile value for mode='percentile'. (e.g. 99.9, 99.99, 99.999, 99.9999)")
    args = parser.parse_args()
    return args


def main(args):
    # Load the model
    print("==> Load ONNX model..")
    onnx_net = onnx.load(args.model)

    # Get input shape
    batch_size = args.batch_size
    imgsz = args.img_size
    input_shape = [batch_size, 3 , imgsz[0], imgsz[1]]


    data_processor = ModelData(input_shape)
    preprocess_args = data_processor.get_preprocessor_args()    

    # Prepare calibration dataset
    print(f"==> Calibration images: {args.calib_table}")
    batcher = ImageBatcher(
        args.calib_data,
        **preprocess_args,
        max_num_images=args.num_images)

    loader = batcher.get_batch()

    num_images = min(batcher.num_images, args.num_images)
    if batcher.num_images < args.num_images:
        print(f"Provided num_images is bigger than the number of calibration images.")
        print(f"Only {num_images} images will be used.")

    if args.mode in ["high", "medium", "low"]:
        if args.mode == "high":
            w_type = ("x330_w_tight", 0)
            a_type = ("x330_tight", 0)
        elif args.mode == "medium":
            w_type = ("x330_w_const", 4)
            a_type = ("x330_const", 4)
        elif args.mode == "low":
            w_type = ("x330_w_relaxed", 3)
            a_type = ("x330_relaxed", 3)
        else:
            w_type = ("x330_w_const", 4)
            a_type = ("x330_const", 4)

        # Do calibration
        calibrations = calibration(onnx_net,
                batch_size,
                num_images,
                loader,
                a_type[0],
                0,
                a_type[1],
                w_type[0],
                w_type[1]
                )

        # Write calibration table
        make_table(args.calib_table, calibrations, a_type[0])

    else:
        print("Current calibration mode is integer calibration for X220.")

        # Do calibration
        calibrations = calibration(onnx_net,
                batch_size,
                num_images,
                loader,
                args.mode,
                args.percentile,
                )

        # Write calibration table
        make_table(args.calib_table, calibrations, args.mode)


if __name__ == "__main__":
    args = parse_args()
    main(args)
