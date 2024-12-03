import torch
import torch.onnx
from yolo_w_multi_classhead import YOLO_MultiClass
import onnx

# 모델 정의
model = YOLO_MultiClass('yolo11l.pt')
model.load_state_dict(torch.load('fashion_classification_l/best_model.pt', map_location='cpu'), strict=False)
model.eval()  # 모델을 평가 모드로 전환

# 더미 입력 데이터 생성 (모델의 입력 크기에 맞게 조정)
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX 파일로 내보내기
onnx_file_path = "yolo_multiclass.onnx"
try:
    torch.onnx.export(
        model,               # 모델 객체
        dummy_input,         # 더미 입력 데이터
        onnx_file_path,      # 내보낼 ONNX 파일 경로
        export_params=True,  # 모델의 학습된 파라미터를 내보낼지 여부
        opset_version=11,    # ONNX opset 버전
        do_constant_folding=True,  # 상수 폴딩 최적화 적용 여부
        input_names=['input'],     # 입력 텐서의 이름
        output_names=['output'],   # 출력 텐서의 이름
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 동적 축 설정
    )
    print(f"모델이 성공적으로 {onnx_file_path}로 내보내졌습니다.")
except Exception as e:
    print(f"ONNX로 내보내는 과정에서 오류가 발생했습니다: {e}")

# ONNX 모델 로드 및 검증
try:
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델이 성공적으로 검증되었습니다.")
except Exception as e:
    print(f"ONNX 모델 검증 과정에서 오류가 발생했습니다: {e}")