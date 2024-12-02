#!/bin/sh

if [ $# -lt 1 ]; then
    dtype="fp32"
else
    dtype=$1
fi

model="models/yolov8n-pose.onnx"
input="datasets/valid/images/"
batch_size=1
output_json=./outputs_json/output_pose_${dtype}.json
num_images=114
conf_thres=0.7
iou_thres=0.65
output=./output_imgs/output_imgs_pose_${dtype}

if [ "$dtype" = "fp32" ]; then
    python3 onnxruntime_sapeon.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --dtype ${dtype} \
                --val \
                --json ${output_json} \
		--save_img \
                --output $output \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images

elif [ "$dtype" = "nf8" -o "$dtype" = "nf16" ]; then
    calib_table="calib_tables/yolov8s_high.json"
    python3 onnxruntime_sapeon.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --calib_table $calib_table \
                --dtype ${dtype} \
                --val \
                --json ${output_json} \
		--save_img \
                --output $output \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images

else
    calib_table="../common/calib_tables/yolov7_percentile-99.999.txt"
    python3 onnxruntime_sapeon.py \
                --model $model \
                --input $input \
                --batch_size $batch_size \
                --calib_table $calib_table \
                --dtype ${dtype} \
                --val \
                --json ${output_json} \
		--save_img \
                --output $output \
                --conf_thres $conf_thres \
                --iou_thres $iou_thres \
                --num_images $num_images                   
fi                
