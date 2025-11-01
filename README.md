AI Part Flow
================

모델 포팅(SDK flow)
------------------
> 1. validation, clibration을 위한 Custom dataset 준비
> 
> 2. pretrained model을 onnx로 export
> > ultralistic 모듈을 통해 얻은 pt 파일은 NMS가 포함되지 않아 onnx_runtime에서 nms 추가
> 
> 3. ONNX_RUNTIME에서 FP32(CPU), NF16, NF8(NPU) 데이터 유형을 제공하는데, CPU ExecutionProvider과 Sapeon ExecutionProvider를 통해 정확도 검증 가능.
> 
> 4. NPU(Sapeon X330)으로 포팅할 경우 High, Medium, Low CalibMode를 제공하는데, 최적화 정도 <-> 정확도의 수준 차이를 확인 할 수 있고, 준비한 ONNX모델의 블록들에 맞는 Calib Table을 따로 생성한다. 
> 
> 5. Sapeon X330에서 Compile 및 실행하여 ONNX 모델의 정확도 및 inference time등을 확인할 수 있다.
> > Custom dataset을 검증하기 위해 COCOEvaluation 모듈을 사용하는데, Custom dataset을 COCO dataset의 annotation에 맞는 형태의 json 파일로 생성하여 Custom annotation을 생성


AI serving flow
---------------
> 1. 주 서버 에서 AI API server로 Base64 str 형태의 이미지 파일과 함께 Request가 들어오면 jpg 확장자로 AI 서버에 저장
> 
> 2. 이미지를 input으로 두 가지 모델(object, pose)로 inference를 진행
> > 가장 성능이 좋았던 nf16-low mode를 통해 object detection을 진행하고, pose모델의 경우 npu에 포팅이 불가능해 cpu로 inference.
> > 성능은 inference time 0.05s를 treshold로 정하고 가장 mAP가 높게 측정되는 mode를 선택.
> 
> 3. inference 과정에서 image_id, category_id, bbox, score 형태의 json 파일을 저장.
> 
> 4. Fast API에서 inference된 json 정보를 조합하여 object model과 pose 모델에서 중복되는 class를 제거하고 총 10개의 class에 대한 output을 메인 서버로 response
> > 10개의 class는 {Hardhat, No-Hardhat, No-Safety Vest, Person, Safety Vest, machinery, vehicle, Fallperson, Walkwithphone, Sosperson}
