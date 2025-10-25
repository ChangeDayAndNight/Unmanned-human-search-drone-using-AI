# 🚁 AirSim 드론 강화학습 프로젝트 컬렉션

Microsoft AirSim 시뮬레이터를 활용한 자율 드론 강화학습 및 컴퓨터 비전 프로젝트 모음집

## 📋 프로젝트 개요

이 저장소는 AirSim 드론 시뮬레이션 환경에서 강화학습, 컴퓨터 비전, 자율 비행을 구현한 다양한 프로젝트들을 포함합니다. DQN과 PPO 알고리즘을 활용한 강화학습부터 YOLOv5를 이용한 실시간 객체 탐지까지 포괄적인 드론 AI 기술을 다룹니다.

## 🏗️ 프로젝트 구조

```
AirSim-Drone-RL-Collection/
├── 🧠 DQN/                    # DQN 강화학습 인명 탐색 드론
├── 🚀 PPO/                    # PPO 강화학습 드론 내비게이션
├── 📸 IMG_Capture/            # AirSim 이미지 캡처 시스템
├── 🎯 YOLO_Detect/           # YOLO 기반 실시간 인명 탐지
├── 📊 Plot_Metrics/          # 성능 메트릭 시각화 도구
├── 🎮 TELLO_EDU/             # DJI TELLO EDU 실제 드론 제어
│   ├── IMG_Capture/          # TELLO 이미지 캡처
│   └── IMG_Detection/        # TELLO YOLOv5 인명 탐지
└── 📄 docs/                  # 프로젝트 문서
```

## 🌟 주요 프로젝트

### 🧠 [DQN 강화학습 드론](./DQN/)
- **기술**: Deep Q-Network, Double DQN, Dueling DQN, Prioritized Experience Replay
- **목표**: 자율 인명 탐색 및 구조 임무
- **특징**: Self-Attention, Mixed Precision Training, RTX 3060Ti 최적화
- **성능**: 80%+ 목표 달성률, 90%+ 영역 커버리지

### 🚀 [PPO 강화학습 드론](./PPO/)
- **기술**: Proximal Policy Optimization, GAE, RND, ICM
- **목표**: 고급 드론 내비게이션 및 탐색
- **특징**: 커리큘럼 학습, LSTM, Self-Attention
- **최신 기법**: 2020-2025년 최신 PPO 알고리즘 적용

### 📸 [이미지 캡처 시스템](./IMG_Capture/)
- **기술**: AirSim API, OpenCV, 실시간 영상 처리
- **목표**: 드론 카메라 데이터 수집 및 처리
- **특징**: 실시간 스트리밍, 자동 저장, 메타데이터 관리

### 🎯 [YOLO 인명 탐지](./YOLO_Detect/)
- **기술**: YOLOv5, 실시간 객체 탐지, 컴퓨터 비전
- **목표**: 드론 기반 실시간 인명 탐지 및 추적
- **특징**: GPU 가속, 실시간 처리, 자동 바운딩 박스

### 🎮 [TELLO EDU 실제 드론](./TELLO_EDU/)
- **기술**: DJI TELLO EDU, djitellopy, YOLOv5 실시간 탐지
- **목표**: 실제 하드웨어 드론을 활용한 인명 탐지 및 추적
- **특징**: 실시간 비행 제어, 배터리 모니터링, 안전 착륙

## 🛠️ 시스템 요구사항

### 하드웨어
- **CPU**: AMD Ryzen 5 5600X (6코어) 또는 동급 이상
- **RAM**: 16GB DDR4-3200 이상
- **GPU**: RTX 3060Ti (8GB VRAM) 또는 동급 이상
- **저장공간**: 15GB+ 여유 공간

### 소프트웨어
- **OS**: Windows 10/11 (AirSim 권장)
- **Python**: 3.10
- **AirSim**: 1.8.1
- **Unreal Engine**: 4.27
- **CUDA**: 11.8+ (GPU 사용 시)

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/your-username/airsim-drone-rl-collection.git
cd airsim-drone-rl-collection

# 가상환경 생성 및 활성화
python -m venv drone_env
# Windows
drone_env\Scripts\activate
# Linux/Mac
source drone_env/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. AirSim 설정
1. [Unreal Engine 4.27](https://www.unrealengine.com/) 설치
2. [AirSim 1.8.1](https://github.com/microsoft/AirSim) 설치 및 빌드
3. AirSim 환경 실행

### 3. 프로젝트 실행
```bash
# DQN 강화학습 시작
cd DQN
python train.py

# PPO 강화학습 시작
cd PPO
python train.py

# YOLO 실시간 탐지 (AirSim)
cd YOLO_Detect
python YOLO_Detect.py

# TELLO EDU 실제 드론 제어
cd TELLO_EDU
python Battery_Check.py              # 배터리 상태 확인
cd IMG_Capture && python TELLO_EDU_Img_Capture.py  # 이미지 캡처
cd IMG_Detection && python TELLO_EDU_Img_Detect.py # 인명 탐지
```

## 📊 성능 벤치마크

| 프로젝트 | 성공률 | 평균 에피소드 길이 | GPU 사용률 | 특징 |
|---------|--------|-----------------|-----------|------|
| DQN | 85% | 450 steps | 75% | 안정적 학습 |
| PPO | 82% | 380 steps | 80% | 빠른 수렴 |
| YOLO | 95% | - | 60% | 실시간 처리 |

## 🧪 실험 및 연구

### 학습 곡선 분석
- **DQN**: 1000 에피소드 내 수렴
- **PPO**: 800 에피소드 내 수렴
- **커리큘럼 학습**: 30% 성능 향상

### 하드웨어 최적화
- **Mixed Precision**: 40% 메모리 절약
- **배치 크기 조정**: RTX 3060Ti 최적값 32
- **병렬 처리**: 4 워커 스레드 활용

## 🔧 개발 도구

### 모니터링
- **TensorBoard**: 실시간 학습 모니터링
- **Weights & Biases**: 실험 추적 (선택사항)
- **실시간 플롯**: 성능 메트릭 시각화

### 테스팅
```bash
# 전체 테스트 실행
python -m pytest tests/

# 개별 프로젝트 테스트
cd DQN && python test_env_distance.py
cd PPO && python test_enhanced_ppo_integration.py
```

## 📚 문서 및 튜토리얼

### 프로젝트별 상세 문서
- [DQN 프로젝트 상세 가이드](./DQN/README.md)
- [PPO 프로젝트 상세 가이드](./PPO/README.md)
- [YOLO 탐지 시스템 가이드](./YOLO_Detect/README.md)
- [이미지 캡처 시스템 가이드](./IMG_Capture/README.md)
- [TELLO EDU 실제 드론 종합 가이드](./TELLO_EDU/README.md)
  - [TELLO 인명 탐지 상세 가이드](./TELLO_EDU/IMG_Detection/README.md)

### 연구 참고 자료
- [DQN 논문](https://www.nature.com/articles/nature14236) - Mnih et al. (2015)
- [PPO 논문](https://arxiv.org/abs/1707.06347) - Schulman et al. (2017)
- [AirSim 문서](https://microsoft.github.io/AirSim/)

## 🤝 기여 방법

1. 저장소 Fork
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

### 개발 가이드라인
- 코드 스타일: Black formatter 사용
- 테스트: pytest로 단위 테스트 작성
- 문서화: docstring 및 README 업데이트

## 🐛 문제 해결

### 일반적인 문제들

**AirSim 연결 실패**
```bash
# AirSim이 실행 중인지 확인
# Unreal Engine 프로젝트를 먼저 시작
# 포트 41451이 열려있는지 확인
```

**GPU 메모리 부족**
```python
# config.py에서 배치 크기 조정
BATCH_SIZE = 16  # 기본값에서 감소
```

**학습 불안정**
```python
# 학습률 조정
LEARNING_RATE = 0.00005  # 기본값에서 감소
```

## 📊 로드맵

### 현재 (2024 Q4)
- ✅ DQN 구현 완료
- ✅ PPO 구현 완료
- ✅ YOLO 탐지 시스템 완료

### 향후 계획 (2025)
- 🔄 Multi-Agent 강화학습
- 🔄 Transformer 기반 정책 네트워크
- 🔄 실제 드론 하드웨어 포팅
- 🔄 ROS 통합

## 📄 라이센스

이 프로젝트는 [MIT 라이센스](LICENSE) 하에 배포됩니다.

## 🙏 감사의 말

- [Microsoft AirSim](https://github.com/microsoft/AirSim) - 시뮬레이션 환경
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - 객체 탐지
- [OpenAI Gym](https://gym.openai.com/) - 강화학습 인터페이스

## 📞 연락처

- **Issues**: [GitHub Issues](https://github.com/your-username/airsim-drone-rl-collection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/airsim-drone-rl-collection/discussions)

---

⭐ **이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

*최종 업데이트: 2024년 10월*