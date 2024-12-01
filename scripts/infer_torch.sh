#!/bin/sh

###### Arguments ######
weights="../../common/models/pytorch/yolov7.pt"
annot="../../common/datasets/coco/annotations/instances_mlperf.json"
data="data/mlperf.yaml"
task="test"
img=640
batch=1
conf=0.35
iou=0.65
device="cpu"
name="mlperf_test"
save_json="--save-json"

###### Do not change the option below ######
cd yolov7
python3 infer_torch.py \
        --weights $weights \
        --annot $annot \
        --data $data \
        $save_json \
        --task $task \
        --img $img \
        --batch $batch \
        --conf $conf \
	--iou $iou \
	--device $device \
	--name $name

