import argparse
import json
import math
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from src import ModelData, ImageBatcher, Postprocessor, non_max_suppression
from utils.utils import myconverter, profiling, write_csv, sapeon_device_id
from utils.metrics import evaluate_on_coco


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAPEON SDK - Infer image(s) using onnxruntime SAPEON.")
    parser.add_argument(
        "--model",
        default="models/yolov8-object.onnx",
        type=str,        
        help="Path to the target model in '*.onnx' format.")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/test/images",
        help="Path to validation dataset.")           
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
        "--calib_table",
        type=str,
        default="./calib_tables/yolov8s_high.json",
        help="Path of a calibration table. Use a result of `calib.py`")      
    parser.add_argument(
        "--class_name",
        default="datasets/annotations/class.names",
        type=str,
        help="The path of label list.")    
    parser.add_argument(
        "--dtype",
        default="nf8",
        type=str,
        choices=["fp32", "uint8", "sint8", "sint16", "nf8", "nf16"],
        help="Data type for inference. ['uint8', 'sint8', 'sint16'] is for X220, and ['nf8', 'nf16'] is for X330, ['fp32'] is for CPU.")                   
    parser.add_argument(
        "--save_img",
        action="store_true",
        help="Save outputs image(s).")
    parser.add_argument(
        "--val",
        action="store_true",
        help="Save outputs json for validation.")
    parser.add_argument(
        "--output",
        default="./output_imgs",
        type=str,
        help="Path to outputs image(s).")
    parser.add_argument(
        "--json",
        default="output_json/outputs.json",
        type=str,
        help="Output json file for validation.")
    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
        help="Device ID of SAPEON card. Please check '/dev/aix# (X220)' or '/dev/snx3-# (X330)'.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="The number of image(s) used for inference.")
    parser.add_argument(
        "--annotation",
        default="datasets/annotations/instances_mlperf.json",
        type=str,
        help="Annotation path.")        
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.35,
        help="Object confidence threshold.")
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.65,
        help="IOU threshold for NMS.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Do profiling.")
    args = parser.parse_args()
    return args


def main(args):
    if args.dtype == 'uint8' or args.dtype == 'sint8' or args.dtype == 'sint16':
        device = 'x220'
    else:
        device = 'x330'
    if not args.dtype == 'fp32':
        device_id = args.device_id if args.device_id else sapeon_device_id(device=device)
    if args.profile:
        profiling(device=device)
        args.num_images = 1

    # Get input shape
    batch_size = args.batch_size
    imgsz = args.img_size
    input_shape = [batch_size, 3, imgsz[0], imgsz[1]]

    import onnx
    import onnxruntime as ort

    print("==> Load ONNX model..")
    onnx_net = onnx.load(args.model)
    num_of_runs = math.ceil(args.num_images / batch_size)

    # Set execution provider & dtype
    print(f"==> Data Type: {args.dtype.upper()}")
    if args.dtype == "fp32":
        EP_list = ["CPUExecutionProvider"]
        provider_options = [{}]
    else:
        EP_list = ["SapeonExecutionProvider", "CPUExecutionProvider"]  # SapeonExecutionProvider
        provider_options = [{
            "expected_batch":
                batch_size,
            "calibration_table":
                args.calib_table,
            "device_type":
                "x220" if args.dtype in ["uint8", "sint8", "sint16"] else "x330",
            "data_type":
                args.dtype,
            "device_id":
                device_id
        }, {}]

    # Create inference session
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 2
    ort_session = ort.InferenceSession(
        onnx_net.SerializeToString(),
        session_options,
        providers=EP_list,
        provider_options=provider_options)
    input_name = ort_session.get_inputs()[0].name
    ort_output_names = [x.name for x in ort_session.get_outputs()]

    # Set data loader
    data_processor = ModelData(input_shape, args.class_name, 
                            args.conf_thres, args.iou_thres)
    preprocess_args = data_processor.get_preprocessor_args()    
    postprocessor_args = data_processor.get_postprocessor_args()        
    labels = data_processor.get_postprocessor_args()["classes"]
    letterbox = data_processor.get_preprocessor_args()["letterbox"]
    batcher = ImageBatcher(
        args.input,
        **preprocess_args,
        max_num_images=args.num_images)
    postprocessor = Postprocessor(
        input_shape,
        )

    cnt = 0
    val_data_list = []
    latency_list = []
    start = time.time()
    # Inference
    with tqdm(total=batcher.num_images) as pbar:
        for img_arr_list, [raw_img_arr_list,
                           img_path_list] in batcher.get_batch():
            t0 = time.time()
            ort_input = ort_session.run(ort_output_names,
                                        {input_name: img_arr_list})
            latency_list.append(time.time() - t0)
            ort_outs = dict(zip(ort_output_names, ort_input))

            detections = []
            for k, v in ort_outs.items():
                if k in ort_output_names:
                    detections.append(v)

            # Parse detection outputs
            det = non_max_suppression(
                detections[0],
                **postprocessor_args)

            log_dict = {}
            for i, img in enumerate(raw_img_arr_list):
                cnt += 1
                im0 = img.copy()
                origin_height, origin_width, _ = im0.shape
                resolution_raw = (origin_width, origin_height)
                img_path = img_path_list[i]
                if len(det):
                    boxes, classes, scores = postprocessor.parse_detection_output(
                        det[i], resolution_raw, imgsz, letterbox)

                    # Save outputs json for validation
                    if args.val:
                        id = ".".join(os.path.basename(img_path).split(".")[:-1])
                        for j in range(0, len(classes)):
                            val_data = {}
                            val_data['image_id'] = id
                            val_data['category_id'] = classes[j]
                            val_data['bbox'] = boxes[j]
                            val_data['score'] = float(scores[j])
                            val_data_list.append(val_data)

                    # Save outputs image
                    if args.save_img:
                        obj_detected_img = postprocessor.draw_bboxes(
                            img_path, boxes, scores, classes, labels)
                        os.makedirs(args.output, exist_ok=True)
                        output_path = os.path.join(args.output,
                                                   os.path.basename(img_path))
                        obj_detected_img.save(output_path)

                log_dict['inference_time'] = f"{float(latency_list[-1] * 1000):.4f} ms"
                pbar.set_postfix(log_dict)
                pbar.update(1)

                if cnt >= batcher.num_images:
                    break

            if cnt >= batcher.num_images:
                break
    exc_time = time.time() - start

    if args.val:
        with open(args.json, "w") as fp:
            json.dump(val_data_list, fp, default=myconverter)
        val_results = evaluate_on_coco(args, args.annotation)

    latency = round(float((sum(latency_list) * 1000) / (len(latency_list) * batch_size)),4)
    throughput = round((len(latency_list)*batch_size)/sum(latency_list),2)
    print("===============Results===============")
    print(f"Model: {args.model}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Number of images: {batcher.num_images}")
    print(f"Number of runs: {len(latency_list)}")
    print(
        f"Latency = {latency:.4f} ms"
    )
    print(f"Total Inference Time = {float(sum(latency_list)):.4f} s")
    print(f"Execution Time = {float(exc_time):.4f} s")
    print(f"Throughput = {throughput:.2f} FPS")
    if args.val:
        print(f"mAP: {val_results}")
    print("==> Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
