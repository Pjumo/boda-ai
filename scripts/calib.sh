#!/bin/bash

if [ $# -lt 1 ]; then
    device="X330"
else
    device=$1
fi

echo "============================"
echo "Target Device: ${device}    "
echo "============================"

model="./models/yolov8s-object-100.onnx"
calib_data="./datasets/valid/images"
batch_size=1
num_images=114

if [[ "$device" =~ "330" ]]; then
    mode="low"
    calib_table="./calib_tables/yolov8s_low.json"
    python3 calib.py \
            --model $model \
            --batch_size $batch_size \
            --calib_data $calib_data \
            --calib_table $calib_table \
            --num_images $num_images \
            --mode $mode
else
    mode="percentile"
    percentile=99.999
    calib_table="./calib_tables/yolov7_percentile-99.999.txt"
    python3 calib.py \
            --model $model \
            --batch_size $batch_size \
            --calib_data $calib_data \
            --calib_table $calib_table \
            --num_images $num_images \
            --mode $mode \
            --percentile $percentile
fi