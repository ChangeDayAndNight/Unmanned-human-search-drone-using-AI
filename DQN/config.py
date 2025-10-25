"""
DQN AirSim Drone Configuration
하드웨어 환경: RTX 3060Ti (8GB VRAM), 16GB RAM, AMD Ryzen 5 5600X
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

class Config:
    # Environment Settings
    AIRSIM_HOST = "127.0.0.1"
    AIRSIM_PORT = 41451
    VEHICLE_NAME = "Drone1"

    # Image Settings (RTX 3060Ti 메모리 최적화)
    IMAGE_WIDTH = 224  # 추가 메모리 최적화
    IMAGE_HEIGHT = 168  # 추가 메모리 최적화
    IMAGE_CHANNELS = 3
    IMAGE_TYPE = 0  # Scene RGB

    # State Space (최적화됨)
    STATE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS + 8  # RGB + position vector
    POSITION_DIM = 8  # x, y, z, qz, qw, rel_x, rel_y, rel_z (최적화된 8차원)

    # Action Space (Discrete)
    ACTION_SIZE = 10  # [상승, 하강, 좌, 우, 전진, 후진, 전진-좌, 전진-우, 후진-좌, 후진-우]
    ACTIONS = {
        0: [0, 0, -2],     # 상승 (속도 감소로 정밀도 향상)
        1: [0, 0, 2],      # 하강 (속도 감소로 정밀도 향상)
        2: [0, -3, 0],     # 좌
        3: [0, 3, 0],      # 우
        4: [3, 0, 0],      # 전진
        5: [-3, 0, 0],     # 후진
        6: [2, -2, 0],     # 전진-좌 (대각선, 속도 조정)
        7: [2, 2, 0],      # 전진-우 (대각선, 속도 조정)
        8: [-2, -2, 0],    # 후진-좌 (대각선, 속도 조정)
        9: [-2, 2, 0]      # 후진-우 (대각선, 속도 조정)
    }

    # Movement Settings (속도 최적화 강화)
    MAX_SPEED = 5.0
    MOVEMENT_DURATION = 0.75  # 드론 움직임 지속 시간

    # Action Selection Timing (행동 선택 속도 제어)
    MIN_ACTION_INTERVAL = 0.15  # 행동 간 최소 간격 (150ms)
    ACTION_PROCESSING_TIME = 0.05  # 행동 처리 시간 (50ms)

    # Environment Bounds
    MAP_BOUNDS = {
        'x_min': -100, 'x_max': 100,
        'y_min': -100, 'y_max': 100,
        'z_min': -50, 'z_max': -2  # AirSim에서 z는 음수 (높이)
    }

    # Episode Settings
    MAX_EPISODE_STEPS = 1000
    TARGET_POSITIONS = [
        [20, 20, -10],   # 목표 인명 위치들 (예시)
        [-30, 40, -15],
        [50, -20, -8],
        [-10, -50, -12]
    ]
    TARGET_RADIUS = 5.0  # 목표 도달 판정 반경

    # Reward Function Parameters (재설계됨 - 균형잡힌 보상 체계)
    REWARDS = {
        # 핵심 목표 달성 보상
        'target_reached': 100.0,      # 목표 도달 (최고 보상)

        # 거리 기반 보상 (스케일링 개선)
        'getting_closer': 2.0,        # 목표에 가까워짐 (강화)
        'getting_farther': -3.0,      # 목표에서 멀어짐 (대폭 강화)

        # 치명적 상황 패널티
        'collision': -100.0,          # 충돌
        'out_of_bounds': -50.0,       # 경계 이탈 (약간 완화)

        # 기본 생존 보상
        'step_reward': 0.01,          # 기본 생존 보상 (time_penalty 대체)
        'altitude_penalty': -2.0,     # 고도 위반 강화

        # 행동 다양성 보상 (재조정)
        'movement_reward': 0.1,       # 움직임 보상 (증가)
        'stop_penalty': -0.5,         # 연속 정지 패널티 (강화)

        # 탐색 보상 (조건부로 변경)
        'exploration_bonus': 0.5,     # 새로운 영역 탐색시에만

        # 효율성 보상
        'progress_bonus': 1.0,        # 목표를 향한 진전시
        'efficiency_bonus': 5.0,      # 빠른 목표 도달시
    }

    # DQN Hyperparameters (1000 에피소드 확장 학습) - 최신 기법 최적화
    LEARNING_RATE = 0.0001  # Categorical DQN + Curiosity에 맞게 낮춤
    GAMMA = 0.99  # 표준값으로 조정 (Categorical DQN 안정성)

    # Epsilon scheduling (행동 다양성 보장)
    EPSILON_START = 0.9  # 초기 exploration 강화
    EPSILON_END = 0.05   # 최소 exploration 보장
    EPSILON_DECAY = 0.998  # 더 천천히 감소

    # Memory Settings (RTX 3060Ti 8GB 최적화)
    REPLAY_BUFFER_SIZE = 10000  # 메모리 사용량 대폭 감소 (30000 -> 10000)
    BATCH_SIZE = 8   # 메모리 안전성 확보 (16 -> 8)
    MIN_REPLAY_SIZE = 800   # 더 빠른 학습 시작 (1500 -> 800)

    # Network Update Settings (최신 기법 최적화)
    TARGET_UPDATE_FREQUENCY = 200   # 더 자주 업데이트 (안정성)
    SAVE_FREQUENCY = 50   # 더 자주 저장 (안전성)

    # Prioritized Experience Replay (Categorical DQN 최적화)
    PER_ALPHA = 0.7  # 우선순위 강화
    PER_BETA_START = 0.5  # 더 높은 시작값
    PER_BETA_END = 1.0
    PER_EPS = 1e-6

    # Double DQN
    USE_DOUBLE_DQN = True

    # Dueling DQN
    USE_DUELING_DQN = True

    # Network Architecture (메모리 최적화)
    CNN_FILTERS = [32, 64]  # 메모리 효율성 유지
    CNN_KERNELS = [8, 4]
    CNN_STRIDES = [4, 2]

    HIDDEN_SIZES = [256, 128]  # MLP hidden layers 속도 최적화

    # Training Settings
    NUM_EPISODES = 1000  # 확장된 심화 학습
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"

    # Device Settings
    DEVICE = "cuda"  # RTX 3060Ti 사용
    NUM_WORKERS = 4  # Ryzen 5 5600X 6코어 중 4개 사용

    # Random Seed for Reproducibility
    RANDOM_SEED = 42

    # Logging (1000 에피소드 최적화)
    LOG_INTERVAL = 50   # 1000 에피소드용 로깅 주기
    EVAL_INTERVAL = 200 # 1000 에피소드용 평가 주기
    EVAL_EPISODES = 5   # 더 정확한 평가

    # 2024-2025 최신 기법 적용
    # Noisy Networks (활성화)
    USE_NOISY_NETS = True  # 최신 기법 적용

    # Rainbow DQN components (최적화)
    USE_CATEGORICAL_DQN = False  # 메모리 절약을 위해 비활성화
    CATEGORICAL_ATOMS = 21  # 원자 수 감소로 메모리 절약
    V_MIN = -20.0  # 값 범위 축소
    V_MAX = 20.0   # 값 범위 축소

    USE_MULTI_STEP = True
    MULTI_STEP_N = 3

    # Advanced exploration (단순화)
    USE_UCB_EXPLORATION = False  # Noisy Networks와 충돌 방지를 위해 비활성화
    UCB_C = 2.0  # UCB 상수

    # Intrinsic motivation (최적화)
    USE_CURIOSITY = False  # 메모리 절약을 위해 비활성화
    CURIOSITY_BETA = 0.1  # 계수 감소

    # Advanced regularization (단순화)
    USE_DROPOUT_SCHEDULING = False  # 단순화를 위해 비활성화
    INITIAL_DROPOUT = 0.2  # 고정 드롭아웃만 사용
    FINAL_DROPOUT = 0.2

    # Memory optimization
    GRADIENT_ACCUMULATION_STEPS = 1
    MIXED_PRECISION = True  # RTX 3060Ti에서 메모리 절약

    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(f"{cls.LOG_DIR}/tensorboard", exist_ok=True)

    @classmethod
    def get_state_shape(cls):
        """상태 공간 shape 반환"""
        return (cls.IMAGE_CHANNELS, cls.IMAGE_HEIGHT, cls.IMAGE_WIDTH)

    @classmethod
    def validate_config(cls) -> bool:
        """설정 값들의 유효성 검증 (개선된 버전)"""
        try:
            # 기본 검증
            assert cls.BATCH_SIZE <= cls.REPLAY_BUFFER_SIZE, f"BATCH_SIZE ({cls.BATCH_SIZE}) must be <= REPLAY_BUFFER_SIZE ({cls.REPLAY_BUFFER_SIZE})"
            assert cls.MIN_REPLAY_SIZE <= cls.REPLAY_BUFFER_SIZE, f"MIN_REPLAY_SIZE ({cls.MIN_REPLAY_SIZE}) must be <= REPLAY_BUFFER_SIZE ({cls.REPLAY_BUFFER_SIZE})"
            assert cls.EPSILON_END <= cls.EPSILON_START, f"EPSILON_END ({cls.EPSILON_END}) must be <= EPSILON_START ({cls.EPSILON_START})"
            assert 0 < cls.GAMMA <= 1, f"GAMMA ({cls.GAMMA}) must be in (0, 1]"
            assert cls.PER_ALPHA >= 0, f"PER_ALPHA ({cls.PER_ALPHA}) must be >= 0"
            assert 0 <= cls.PER_BETA_START <= cls.PER_BETA_END <= 1, f"PER_BETA values must be in [0, 1]: START={cls.PER_BETA_START}, END={cls.PER_BETA_END}"

            # 추가 검증
            assert cls.ACTION_SIZE > 0, f"ACTION_SIZE ({cls.ACTION_SIZE}) must be positive"
            assert len(cls.ACTIONS) == cls.ACTION_SIZE, f"ACTIONS dict length ({len(cls.ACTIONS)}) must match ACTION_SIZE ({cls.ACTION_SIZE})"
            assert cls.IMAGE_WIDTH > 0 and cls.IMAGE_HEIGHT > 0, f"Image dimensions must be positive: {cls.IMAGE_WIDTH}x{cls.IMAGE_HEIGHT}"
            assert cls.NUM_EPISODES > 0, f"NUM_EPISODES ({cls.NUM_EPISODES}) must be positive"
            assert cls.LEARNING_RATE > 0, f"LEARNING_RATE ({cls.LEARNING_RATE}) must be positive"

            # 메모리 사용량 예상 검증
            estimated_memory_gb = cls._estimate_memory_usage()
            if estimated_memory_gb > 7.5:  # RTX 3060Ti 8GB 여유분 고려
                logging.warning(f"Estimated memory usage ({estimated_memory_gb:.1f}GB) may exceed GPU memory")

            print("Configuration validation passed!")
            return True

        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during validation: {e}")
            return False

    @classmethod
    def _estimate_memory_usage(cls) -> float:
        """대략적인 GPU 메모리 사용량 추정 (GB)"""
        # 배치 크기 기준 이미지 텐서 메모리
        image_memory = cls.BATCH_SIZE * cls.IMAGE_CHANNELS * cls.IMAGE_HEIGHT * cls.IMAGE_WIDTH * 4  # float32

        # 네트워크 파라미터 메모리 (대략적 추정)
        cnn_params = sum(cls.CNN_FILTERS) * 1000  # 대략적
        mlp_params = sum(cls.HIDDEN_SIZES) * 1000
        network_memory = (cnn_params + mlp_params) * 4  # float32

        # 리플레이 버퍼 메모리
        replay_memory = cls.REPLAY_BUFFER_SIZE * (image_memory / cls.BATCH_SIZE + cls.POSITION_DIM * 4)

        total_bytes = image_memory + network_memory + replay_memory
        return total_bytes / (1024 ** 3)  # Convert to GB

if __name__ == "__main__":
    Config.validate_config()
    Config.create_directories()
    print("Configuration loaded successfully!")