"""
최적화된 메모리 관리자
DQN 학습 중 메모리 효율성을 극대화하는 전용 클래스
"""

import torch
import gc
import psutil
import numpy as np
from typing import Dict, Optional
import time

class OptimizedMemoryManager:
    """메모리 사용량 최적화 및 관리"""

    def __init__(self, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage  # 최대 메모리 사용률
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30.0  # 30초마다 정리

    def get_memory_stats(self) -> Dict[str, float]:
        """현재 메모리 사용 통계"""
        stats = {}

        # CPU 메모리
        memory = psutil.virtual_memory()
        stats['cpu_memory_used_gb'] = memory.used / (1024**3)
        stats['cpu_memory_total_gb'] = memory.total / (1024**3)
        stats['cpu_memory_percent'] = memory.percent

        # GPU 메모리
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            stats['gpu_memory_max_mb'] = torch.cuda.max_memory_allocated() / (1024**2)

        return stats

    def auto_cleanup(self, force: bool = False) -> bool:
        """자동 메모리 정리"""
        current_time = time.time()

        if not force and (current_time - self.last_cleanup_time) < self.cleanup_interval:
            return False

        # 메모리 사용률 체크
        memory_percent = psutil.virtual_memory().percent

        if force or memory_percent > (self.max_memory_usage * 100):
            self.aggressive_cleanup()
            self.last_cleanup_time = current_time
            return True

        return False

    def aggressive_cleanup(self):
        """적극적인 메모리 정리"""
        # Python 가비지 컬렉션
        for _ in range(3):
            gc.collect()

        # GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def optimize_replay_buffer(self, buffer_data: list, max_size: int) -> list:
        """리플레이 버퍼 최적화"""
        if len(buffer_data) > max_size:
            # 최근 데이터 우선 유지, 오래된 데이터 제거
            return buffer_data[-max_size:]
        return buffer_data

    def check_memory_health(self) -> Dict[str, bool]:
        """메모리 상태 건강도 체크"""
        stats = self.get_memory_stats()
        health = {}

        # CPU 메모리 체크
        health['cpu_memory_ok'] = stats['cpu_memory_percent'] < 85.0

        # GPU 메모리 체크
        if torch.cuda.is_available():
            gpu_usage_percent = (stats['gpu_memory_allocated_mb'] /
                               (torch.cuda.get_device_properties(0).total_memory / (1024**2))) * 100
            health['gpu_memory_ok'] = gpu_usage_percent < 85.0
        else:
            health['gpu_memory_ok'] = True

        health['overall_ok'] = all(health.values())
        return health

    def suggest_optimizations(self) -> list:
        """메모리 최적화 제안"""
        suggestions = []
        stats = self.get_memory_stats()

        if stats['cpu_memory_percent'] > 80:
            suggestions.append("CPU 메모리 사용률이 높습니다. 배치 크기를 줄이는 것을 고려하세요.")

        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            gpu_usage = (stats['gpu_memory_allocated_mb'] / gpu_total) * 100

            if gpu_usage > 80:
                suggestions.append("GPU 메모리 사용률이 높습니다. Mixed Precision 학습을 활성화하세요.")

        return suggestions

class OptimizedDataLoader:
    """메모리 효율적인 데이터 로더"""

    def __init__(self, batch_size: int = 32, prefetch_factor: int = 2):
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

    def create_batch_generator(self, data: list):
        """메모리 효율적인 배치 생성"""
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            yield self._optimize_batch(batch)

    def _optimize_batch(self, batch: list):
        """배치 데이터 최적화"""
        # 불필요한 메모리 복사 방지
        return batch

class PerformanceOptimizer:
    """성능 최적화 도구"""

    @staticmethod
    def optimize_torch_settings():
        """PyTorch 최적화 설정"""
        torch.backends.cudnn.benchmark = True  # cuDNN 자동 최적화
        torch.backends.cudnn.deterministic = False  # 성능 우선

        # 메모리 할당자 최적화
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # GPU 메모리 90% 사용

    @staticmethod
    def enable_mixed_precision():
        """Mixed Precision 활성화 체크"""
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            # Tensor Core 지원 여부 (7.0 이상)
            return device_capability[0] >= 7
        return False

    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """추론용 모델 최적화"""
        model.eval()

        # JIT 컴파일 (선택적)
        if hasattr(torch, 'jit'):
            try:
                # 샘플 입력으로 JIT 컴파일 시도
                return torch.jit.optimize_for_inference(model)
            except (RuntimeError, AttributeError, TypeError) as e:
                # JIT 컴파일 실패 시 원본 모델 반환 (일부 모델은 JIT 지원하지 않음)
                pass

        return model

# 전역 인스턴스
memory_manager = OptimizedMemoryManager()
performance_optimizer = PerformanceOptimizer()

if __name__ == "__main__":
    # 메모리 관리자 테스트
    print("메모리 최적화 도구 테스트")

    # 현재 메모리 상태
    stats = memory_manager.get_memory_stats()
    print("현재 메모리 상태:", stats)

    # 건강도 체크
    health = memory_manager.check_memory_health()
    print("메모리 건강도:", health)

    # 최적화 제안
    suggestions = memory_manager.suggest_optimizations()
    if suggestions:
        print("최적화 제안:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("현재 메모리 상태가 양호합니다.")

    # PyTorch 최적화 설정
    performance_optimizer.optimize_torch_settings()
    print("PyTorch 최적화 설정 완료")

    # Mixed Precision 지원 여부
    if performance_optimizer.enable_mixed_precision():
        print("Mixed Precision 학습 지원됨")
    else:
        print("Mixed Precision 학습 미지원")