#!/bin/sh

if [ $# -lt 1 ]; then
    dtype="fp32"
else
    dtype=$1
fi

model="../common/models/onnx/yolov7.onnx"
input="../common/datasets/coco/images/mlperf_calibration"
batch_size=1
num_images=500
conf_thres=0.35
iou_thres=0.65
num_threads=24

if [ "$dtype" = "fp32" ]; then
    python3 onnxruntime_sapeon_multi-threading.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --dtype ${dtype} \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images \
                --num_threads $num_threads

elif [ "$dtype" = "nf8" -o "$dtype" = "nf16" ]; then
    calib_table="../common/calib_tables/yolov7_medium.json"
    python3 onnxruntime_sapeon_multi-threading.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --calib_table $calib_table \
                --dtype ${dtype} \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images \
                --num_threads $num_threads --val           

else
    calib_table="./common/calib_tables/yolov7_percentile-99.999.txt"
    python3 onnxruntime_sapeon_multi-threading.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --calib_table $calib_table \
                --dtype ${dtype} \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images \
                --num_threads $num_threads                                 
fi                
