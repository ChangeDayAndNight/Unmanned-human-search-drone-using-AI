# AirSim YOLOv5 실시간 객체 탐지 드론 시스템

## 개요
AirSim 환경에서 YOLOv5 모델을 사용하여 실시간 객체 탐지를 수행하는 드론 제어 시스템입니다.

## 주요 기능
- YOLOv5n 모델을 사용한 실시간 객체 탐지
- 드론 자동 이동 및 위치 제어
- OpenCV를 통한 실시간 영상 출력
- 탐지된 객체에 바운딩 박스 및 라벨 표시
- 실시간 드론 위치 및 탐지 통계 오버레이

## 시스템 요구사항
- Windows 10/11
- Python 3.10+
- NVIDIA GPU (RTX 3060Ti 권장)
- AirSim 1.8.1 + Unreal Engine 4.27
- 8GB+ GPU 메모리

## 설치 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. AirSim 설정
Unreal Engine에서 AirSim 환경을 실행해야 합니다.

### 3. YOLOv5 모델 확인
모델 경로가 올바른지 확인:
```
C:/Users/cktmd/Desktop/yolov5/runs/train/exp6/weights/best.pt
```

## 사용 방법

### 1. 시스템 실행
```bash
python yolo_drone_controller.py
```

### 2. 드론 동작 순서
1. **모델 로딩**: YOLOv5 모델 자동 로드
2. **AirSim 연결**: 드론 연결 및 제어 활성화
3. **이륙 및 이동 시작**: 드론이 이륙 후 목표 좌표 (X=0, Y=200, Z=-25)로 이동 시작
4. **실시간 탐지**: 이동과 동시에 카메라 영상 표시 및 객체 탐지 시작
5. **이동 중 실시간 피드백**: 목표 지점까지의 거리와 이동 상태 실시간 표시

### 3. 제어 키
- `Q`: 프로그램 종료

## 파일 구조
```
IMG_Capture/
├── yolo_drone_controller.py    # 메인 실행 파일
├── IMG_Capture_Code.py        # 기존 이미지 캡처 코드
├── requirements.txt           # 의존성 목록
└── README.md                 # 이 파일
```

## 코드 주요 기능

### YOLODroneController 클래스
- **__init__()**: YOLOv5 모델 로드 및 AirSim 연결
- **move_to_target()**: 드론을 지정 좌표로 이동
- **get_camera_image()**: 드론 카메라에서 실시간 이미지 획득
- **detect_objects()**: YOLOv5를 사용한 객체 탐지
- **run_real_time_detection()**: 실시간 영상 스트리밍 및 탐지

### 실시간 정보 표시
- 현재 프레임 수
- 드론 위치 (X, Y, Z 좌표)
- 현재 프레임의 탐지 객체 수
- 총 탐지된 객체 수

## 설정 변경

### 드론 이동 좌표 변경
`yolo_drone_controller.py`의 main() 함수에서:
```python
controller = YOLODroneController(
    target_x=0,    # X 좌표
    target_y=200,  # Y 좌표
    target_z=-25   # Z 좌표 (고도)
)
```

### YOLOv5 모델 변경
생성자에서 model_path 파라미터 수정:
```python
controller = YOLODroneController(
    model_path="새로운/모델/경로/best.pt"
)
```

### 탐지 임계값 조정
`__init__()` 메서드에서:
```python
self.model.conf = 0.25  # 신뢰도 임계값 (0-1)
self.model.iou = 0.45   # NMS IoU 임계값 (0-1)
```

## 문제 해결

### 1. AirSim 연결 오류
```
Retry connection over the limit
```
- Unreal Engine에서 AirSim 환경이 실행 중인지 확인
- AirSim 포트(41451)가 열려있는지 확인

### 2. YOLOv5 모델 로드 실패
- 모델 파일 경로가 올바른지 확인
- PyTorch 및 torchvision 버전 호환성 확인

### 3. GPU 메모리 부족
- 배치 크기나 이미지 해상도 조정
- 다른 GPU 사용 프로그램 종료

## 성능 최적화
- RTX 3060Ti 기준으로 최적화됨
- Mixed Precision Training 지원
- 실시간 처리를 위한 프레임 제한 (~30 FPS)

## 주의사항
- AirSim 환경이 반드시 실행 중이어야 함
- 드론 조작 시 안전에 유의
- GPU 메모리 사용량 모니터링 권장