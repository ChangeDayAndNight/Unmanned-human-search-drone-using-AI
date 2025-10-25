"""
Utility functions for DQN AirSim Drone project
로깅, 시각화, 성능 모니터링, 메모리 관리 등의 유틸리티
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import time
import psutil
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import tensorboard
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """향상된 로깅 시스템"""

    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_path = os.path.join(log_dir, f"{self.experiment_name}.log")

        # 로거 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(os.path.join(log_dir, "tensorboard", self.experiment_name))

        # 메트릭 저장
        self.metrics_history = []
        self.episode_data = []

        self.logger.info(f"로거 초기화 완료: {self.experiment_name}")

    def log_episode(self, episode: int, stats: Dict):
        """에피소드 결과 로깅"""
        self.logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {stats.get('reward', 0):7.2f} | "
            f"Length: {stats.get('length', 0):4d} | "
            f"Epsilon: {stats.get('epsilon', 0):.3f} | "
            f"Loss: {stats.get('loss', 0):.4f} | "
            f"Targets: {stats.get('targets_found', 0):2d} | "
            f"Coverage: {stats.get('exploration_coverage', 0):.2%}"
        )

        # TensorBoard 로깅
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(f"Episode/{key}", value, episode)

        # 메트릭 저장
        stats['episode'] = episode
        stats['timestamp'] = datetime.now().isoformat()
        self.episode_data.append(stats)

    def log_training_step(self, step: int, metrics: Dict):
        """훈련 스텝 메트릭 로깅"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(f"Training/{key}", value, step)

    def log_model_info(self, model: torch.nn.Module):
        """모델 정보 로깅"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"모델 파라미터 수: {total_params:,}")
        self.logger.info(f"훈련 가능한 파라미터 수: {trainable_params:,}")

        # GPU 메모리 사용량
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(f"GPU 메모리 사용량: {memory_allocated:.2f} MB / {memory_reserved:.2f} MB")

    def save_metrics(self):
        """메트릭을 JSON 파일로 저장"""
        metrics_path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.episode_data, f, indent=2)

    def close(self):
        """로거 종료"""
        self.save_metrics()
        self.tb_writer.close()
        self.logger.info("로거 종료")

class PerformanceMonitor:
    """시스템 성능 모니터링"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.start_time = time.time()

        # 성능 메트릭 저장
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_usage = deque(maxlen=1000)
        self.gpu_memory = deque(maxlen=1000)

    def update(self) -> Dict:
        """성능 메트릭 업데이트"""
        self.step_count += 1

        # CPU/Memory 사용량
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)

        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'elapsed_time': time.time() - self.start_time
        }

        # GPU 메트릭 (사용 가능한 경우)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2

            self.gpu_memory.append(gpu_memory_allocated)
            metrics.update({
                'gpu_memory_allocated': gpu_memory_allocated,
                'gpu_memory_reserved': gpu_memory_reserved
            })

        return metrics

    def get_summary(self) -> Dict:
        """성능 요약 통계"""
        return {
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_gpu_memory': np.mean(self.gpu_memory) if self.gpu_memory else 0,
            'total_steps': self.step_count,
            'total_time': time.time() - self.start_time
        }

class Visualizer:
    """학습 과정 시각화"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_training_curves(self, episode_data: List[Dict], save_name: str = "training_curves"):
        """학습 곡선 플롯"""
        if not episode_data:
            return

        df = pd.DataFrame(episode_data)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('DQN Training Progress', fontsize=16)

        # 에피소드 보상
        if 'reward' in df.columns:
            axes[0, 0].plot(df['episode'], df['reward'], alpha=0.7)
            axes[0, 0].plot(df['episode'], df['reward'].rolling(50).mean(), linewidth=2)
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')

        # 에피소드 길이
        if 'length' in df.columns:
            axes[0, 1].plot(df['episode'], df['length'], alpha=0.7)
            axes[0, 1].plot(df['episode'], df['length'].rolling(50).mean(), linewidth=2)
            axes[0, 1].set_title('Episode Length')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')

        # 발견된 목표 수
        if 'targets_found' in df.columns:
            axes[0, 2].plot(df['episode'], df['targets_found'], alpha=0.7)
            axes[0, 2].plot(df['episode'], df['targets_found'].rolling(50).mean(), linewidth=2)
            axes[0, 2].set_title('Targets Found')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Count')

        # Epsilon 값
        if 'epsilon' in df.columns:
            axes[1, 0].plot(df['episode'], df['epsilon'])
            axes[1, 0].set_title('Exploration Rate (Epsilon)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')

        # 손실 함수
        if 'loss' in df.columns:
            valid_loss = df[df['loss'] > 0]
            if not valid_loss.empty:
                axes[1, 1].plot(valid_loss['episode'], valid_loss['loss'], alpha=0.7)
                axes[1, 1].plot(valid_loss['episode'], valid_loss['loss'].rolling(50).mean(), linewidth=2)
                axes[1, 1].set_title('Training Loss')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].set_yscale('log')

        # 탐색 커버리지
        if 'exploration_coverage' in df.columns:
            axes[1, 2].plot(df['episode'], df['exploration_coverage'], alpha=0.7)
            axes[1, 2].plot(df['episode'], df['exploration_coverage'].rolling(50).mean(), linewidth=2)
            axes[1, 2].set_title('Exploration Coverage')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Coverage %')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_q_values_distribution(self, q_values: List[float], save_name: str = "q_values_dist"):
        """Q 값 분포 시각화"""
        if not q_values:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(q_values, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(q_values), color='red', linestyle='--', label=f'Mean: {np.mean(q_values):.3f}')
        plt.title('Q-Values Distribution')
        plt.xlabel('Q-Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_exploration_heatmap(self, exploration_map: np.ndarray, save_name: str = "exploration_heatmap"):
        """탐색 히트맵 시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(exploration_map, cmap='YlOrRd', cbar=True)
        plt.title('Exploration Heatmap')
        plt.xlabel('Y Position')
        plt.ylabel('X Position')
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def create_training_animation(self, episode_data: List[Dict], save_name: str = "training_animation"):
        """학습 과정 애니메이션 생성"""
        # 애니메이션 구현 (선택사항)
        pass

class MemoryManager:
    """메모리 관리 유틸리티"""

    @staticmethod
    def clear_gpu_cache():
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_memory_usage() -> Dict:
        """현재 메모리 사용량 반환"""
        memory_info = {
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_used_gb': psutil.virtual_memory().used / 1024**3,
            'cpu_memory_total_gb': psutil.virtual_memory().total / 1024**3
        }

        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'gpu_memory_max_mb': torch.cuda.max_memory_allocated() / 1024**2
            })

        return memory_info

    @staticmethod
    def optimize_memory():
        """메모리 최적화"""
        import gc
        gc.collect()
        MemoryManager.clear_gpu_cache()

class ConfigValidator:
    """설정 검증 유틸리티"""

    @staticmethod
    def validate_config(config) -> List[str]:
        """설정 유효성 검사"""
        warnings = []

        # GPU 메모리 체크
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb < 6:
                warnings.append(f"GPU 메모리 부족: {gpu_memory_gb:.1f}GB (권장: 8GB 이상)")

        # 배치 크기 체크
        if config.BATCH_SIZE > config.REPLAY_BUFFER_SIZE // 10:
            warnings.append("배치 크기가 버퍼 크기에 비해 너무 큼")

        # 하이퍼파라미터 범위 체크
        if not (0 < config.GAMMA <= 1):
            warnings.append(f"잘못된 할인 인자: {config.GAMMA}")

        if not (0 <= config.EPSILON_END <= config.EPSILON_START <= 1):
            warnings.append("잘못된 epsilon 설정")

        return warnings

class ModelCheckpointer:
    """모델 체크포인트 관리"""

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(self, agent, episode: int, metrics: Dict):
        """체크포인트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_ep{episode:06d}_{timestamp}.pth"
        filepath = os.path.join(self.save_dir, filename)

        agent.save_model(filepath, episode)

        # 체크포인트 목록 관리
        self.checkpoints.append({
            'filepath': filepath,
            'episode': episode,
            'metrics': metrics,
            'timestamp': timestamp
        })

        # 오래된 체크포인트 삭제
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint['filepath']):
                os.remove(old_checkpoint['filepath'])

    def get_best_checkpoint(self, metric: str = 'reward') -> Optional[str]:
        """최고 성능 체크포인트 반환"""
        if not self.checkpoints:
            return None

        best_checkpoint = max(self.checkpoints,
                            key=lambda x: x['metrics'].get(metric, float('-inf')))
        return best_checkpoint['filepath']

def set_random_seeds(seed: int):
    """모든 랜덤 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # AirSim은 별도의 시드 설정이 필요할 수 있음

def create_experiment_directory(base_dir: str, experiment_name: str = None) -> str:
    """실험 디렉토리 생성"""
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    return exp_dir

def format_time(seconds: float) -> str:
    """시간을 읽기 쉬운 형태로 포맷"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_eta(current_episode: int, total_episodes: int, elapsed_time: float) -> str:
    """남은 시간 추정"""
    if current_episode == 0:
        return "Unknown"

    avg_time_per_episode = elapsed_time / current_episode
    remaining_episodes = total_episodes - current_episode
    eta_seconds = remaining_episodes * avg_time_per_episode

    return format_time(eta_seconds)

if __name__ == "__main__":
    # 유틸리티 테스트
    print("유틸리티 테스트 시작...")

    # 메모리 사용량 확인
    memory_info = MemoryManager.get_memory_usage()
    print("메모리 사용량:", memory_info)

    # 랜덤 시드 설정
    set_random_seeds(42)
    print("랜덤 시드 설정 완료")

    # 실험 디렉토리 생성
    exp_dir = create_experiment_directory("experiments", "test_exp")
    print(f"실험 디렉토리 생성: {exp_dir}")

    # 시간 포맷 테스트
    print(f"포맷된 시간: {format_time(3661)}")  # 1:01:01
    print(f"ETA: {calculate_eta(100, 1000, 3600)}")  # 8시간 추정

    print("유틸리티 테스트 완료!")