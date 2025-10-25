# YOLO AirSim 인명 탐지 시스템

Unreal_Engine.pt YOLOv5n 모델을 사용한 AirSim 드론 실시간 인명 탐지 시스템

## 기능
- AirSim 시뮬레이터에서 드론 자동 제어
- 실시간 카메라 영상 스트리밍
- YOLO 모델을 이용한 인명 탐지
- 목표 좌표 설정 및 자동 이동
- OpenCV를 통한 실시간 영상 표시 (오버레이 없음)

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. AirSim 설정:
- Unreal Engine 4.27 + AirSim 1.8.1 설치 필요
- AirSim 시뮬레이터 실행

## 사용 방법

### 방법 1: 간단한 실행 (권장)
```bash
python run_detector.py
```

### 방법 2: 고급 기능 실행
```bash
python yolo_airsim_detector.py
```

### 방법 3: 비동기 실행
```bash
python simple_yolo_detector.py
```

## 실행 순서

1. **AirSim 시뮬레이터 실행** (Unreal Engine)
2. **프로그램 실행** (위의 방법 중 선택)
3. **목표 좌표 입력**:
   - X 좌표 (미터)
   - Y 좌표 (미터)
   - Z 좌표 (음수값, 고도)
4. **실시간 인명 탐지 시작**
5. **Q 키로 종료**

## 파일 구조
```
YOLO_Detect/
├── Unreal_Engine.pt          # 훈련된 YOLO 모델
├── yolo_airsim_detector.py   # 메인 탐지 시스템
├── run_detector.py           # 간단한 실행 스크립트
├── simple_yolo_detector.py   # 비동기 버전
├── requirements.txt          # 의존성 패키지
└── README.md                # 사용법 안내
```

## 주요 특징

- **실시간 처리**: 드론 이동과 동시에 인명 탐지 수행
- **오버레이 없음**: OpenCV 창에 순수한 영상만 표시
- **자동 이동**: 설정된 목표 좌표로 자동 이동
- **GPU 가속**: PyTorch CUDA 지원으로 빠른 추론
- **안전 착륙**: 프로그램 종료 시 자동 착륙

## 조작법

- **Q 키**: 프로그램 종료 및 드론 착륙
- **Ctrl+C**: 긴급 중단

## 주의사항

1. AirSim 시뮬레이터가 먼저 실행되어야 합니다
2. 목표 Z 좌표는 음수값으로 입력하세요 (예: -10)
3. 프로그램 종료 시 드론이 자동으로 착륙합니다
4. GPU 메모리가 부족한 경우 기본 YOLOv5n 모델을 사용합니다

## 문제 해결

- **AirSim 연결 실패**: 시뮬레이터가 실행 중인지 확인
- **모델 로딩 실패**: Unreal_Engine.pt 파일 존재 확인
- **카메라 영상 없음**: AirSim 카메라 설정 확인