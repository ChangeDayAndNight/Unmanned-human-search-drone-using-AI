# 🚁 DQN AirSim 무인 인명 탐색 드론

AirSim 시뮬레이터 기반의 Deep Q-Network(DQN) 강화학습을 활용한 자율 인명 탐색 드론 프로젝트입니다.

## 📋 프로젝트 개요

본 프로젝트는 Unreal Engine 기반 AirSim 시뮬레이터에서 DQN 강화학습을 통해 드론이 자율적으로 인명을 탐색하도록 학습시키는 시스템입니다.

### 🎯 주요 특징

- **최신 DQN 기법 적용**: Double DQN, Dueling DQN, Prioritized Experience Replay
- **Multi-step Learning**: 3-step 부트스트래핑으로 학습 효율성 향상
- **Mixed Precision Training**: RTX 3060Ti 메모리 최적화
- **실시간 모니터링**: TensorBoard, 성능 메트릭, 시각화
- **견고한 환경 설계**: 충돌 감지, 경계 체크, 보상 함수 최적화

## 🛠️ 시스템 요구사항

### 하드웨어
- **CPU**: AMD Ryzen 5 5600X (6코어) 또는 동급
- **RAM**: 16GB DDR4-3200 이상
- **GPU**: RTX 3060Ti (8GB VRAM) 또는 동급 이상
- **저장공간**: 10GB 이상

### 소프트웨어
- **OS**: Windows 10/11 (AirSim 권장)
- **Python**: 3.10
- **AirSim**: 1.8.1
- **Unreal Engine**: 4.27 (AirSim 호환 버전)

## 🚀 설치 및 설정

### 1. AirSim 설치

```bash
# AirSim 설치 (Windows)
# 1. Unreal Engine 4.27 설치
# 2. AirSim 빌드 및 설치
# 자세한 설치 방법: https://microsoft.github.io/AirSim/
```

### 2. Python 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
venv\\Scripts\\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 3. AirSim 설정

`Documents/AirSim/settings.json` 파일 생성:

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthroughOnMove": false,
      "EnableCollisions": true
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 320,
        "Height": 240,
        "FOV_Degrees": 90
      }
    ]
  },
  "ClockSpeed": 1.0
}
```

## 📂 프로젝트 구조

```
DQN_No_Human/
├── config.py              # 하이퍼파라미터 및 설정
├── environment.py          # AirSim 환경 래퍼
├── network.py             # DQN 네트워크 아키텍처
├── dqn_agent.py           # DQN 에이전트 클래스
├── utils.py               # 유틸리티 함수들
├── train.py               # 메인 학습 스크립트
├── requirements.txt       # 필요 패키지 목록
├── README.md             # 프로젝트 문서
└── experiments/          # 실험 결과 저장소
    ├── logs/            # 학습 로그
    ├── checkpoints/     # 모델 체크포인트
    └── plots/           # 시각화 결과
```

## 🎮 사용법

### 학습 시작

```bash
# 기본 학습
python train.py

# 실험 이름 지정
python train.py --experiment my_experiment

# 체크포인트에서 재시작
python train.py --resume checkpoints/latest_model.pth

# 평가 전용 모드
python train.py --eval_only --resume checkpoints/best_model.pth
```

### 설정 수정

`config.py` 파일에서 주요 설정을 조정할 수 있습니다:

```python
# 주요 하이퍼파라미터
LEARNING_RATE = 0.0001      # 학습률
BATCH_SIZE = 32             # 배치 크기
EPSILON_DECAY = 0.995       # 탐험률 감소
NUM_EPISODES = 2000         # 총 에피소드 수

# 보상 함수 조정
REWARDS = {
    'target_reached': 100.0,     # 목표 도달 보상
    'collision': -100.0,         # 충돌 페널티
    'exploration_bonus': 0.2,    # 탐색 보너스
}
```

## 📊 모니터링

### TensorBoard 실행

```bash
tensorboard --logdir experiments/logs/tensorboard
# http://localhost:6006 에서 확인
```

### 주요 메트릭

- **Episode Reward**: 에피소드별 누적 보상
- **Targets Found**: 발견한 목표 인명 수
- **Exploration Coverage**: 탐색한 영역 비율
- **Training Loss**: 학습 손실 함수
- **Q-Value Distribution**: Q 값 분포

## 🧠 DQN 아키텍처

### 네트워크 구조

1. **CNN Backbone**: 드론 카메라 이미지 처리
   - 3개 컨볼루션 레이어 (32, 64, 64 필터)
   - BatchNorm + ReLU 활성화
   - Adaptive Average Pooling

2. **Position Encoder**: 드론 위치/자세 정보 처리
   - 9차원 입력 (x,y,z,roll,pitch,yaw,target_rel_x,y,z)
   - 2층 MLP (128 → 256)

3. **Feature Fusion**: 이미지와 위치 정보 결합
   - Concatenation + MLP (512 → 256)
   - Self-Attention 메커니즘

4. **Dueling Head**: 가치 함수 분리
   - Value Stream: 상태 가치 V(s)
   - Advantage Stream: 행동 우위 A(s,a)
   - Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

### 최신 기법 적용

- **Double DQN**: 과대추정 방지
- **Prioritized Experience Replay**: 중요한 경험 우선 학습
- **Multi-step Learning**: 3-step 부트스트래핑
- **Mixed Precision**: GPU 메모리 최적화
- **Gradient Clipping**: 학습 안정성 향상

## 🎯 보상 함수

```python
보상 = 목표도달(+100) + 접근보상(+1~+5) + 탐색보너스(+0.2)
     - 충돌페널티(-100) - 경계이탈(-50) - 시간페널티(-0.1)
```

## 📈 성능 최적화

### GPU 메모리 최적화
- 이미지 크기: 640x480 → 320x240
- 배치 크기: 32 (RTX 3060Ti 최적화)
- Mixed Precision Training
- Gradient Accumulation

### CPU 병렬 처리
- 4개 워커 스레드 활용
- 비동기 데이터 수집
- 경험 재생 버퍼 최적화

## 🐛 문제 해결

### 일반적인 문제들

1. **AirSim 연결 실패**
   ```bash
   # AirSim이 실행 중인지 확인
   # settings.json 설정 확인
   # 방화벽 설정 확인
   ```

2. **GPU 메모리 부족**
   ```python
   # config.py에서 배치 크기 줄이기
   BATCH_SIZE = 16  # 기본값 32에서 감소
   ```

3. **학습 불안정**
   ```python
   # 학습률 낮추기
   LEARNING_RATE = 0.00005
   # Replay buffer 크기 늘리기
   REPLAY_BUFFER_SIZE = 100000
   ```

## 📝 실험 가이드

### 하이퍼파라미터 튜닝 순서

1. **학습률**: 0.0001 → 0.00005 → 0.0002
2. **배치 크기**: 32 → 64 → 16
3. **네트워크 크기**: 필터 수, 은닉층 크기 조정
4. **보상 함수**: 각 요소별 가중치 조정

### 성능 지표

- **수렴 속도**: 1000 에피소드 내 안정화 목표
- **탐색 효율**: 90% 이상 영역 커버리지
- **목표 달성률**: 80% 이상 성공률

## 🤝 기여 방법

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [Microsoft AirSim](https://github.com/microsoft/AirSim) - 시뮬레이션 환경
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [OpenAI Gym](https://gym.openai.com/) - 강화학습 인터페이스

## 📚 참고 문헌

1. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature (2015)
2. Van Hasselt, H., et al. "Deep reinforcement learning with double q-learning." AAAI (2016)
3. Wang, Z., et al. "Dueling network architectures for deep reinforcement learning." ICML (2016)
4. Schaul, T., et al. "Prioritized experience replay." ICLR (2016)
5. Fortunato, M., et al. "Noisy networks for exploration." ICLR (2018)

---

**📧 문의사항**: 프로젝트 관련 문의사항은 Issues 탭을 이용해 주세요.

**⭐ Star**: 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!