"""
최적화 학습 스크립트
확장된 학습 시간을 활용한 심화 강화학습
"""

import torch
import numpy as np
import time
import argparse
import signal
import sys
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from curriculum import (
    CurriculumConfig,
    print_curriculum_progress,
    CURRICULUM_MILESTONES,
    get_stage_recommendations
)
from environment import AirSimDroneEnv
from dqn_agent import DQNAgent
from utils import (
    Logger, PerformanceMonitor, Visualizer, MemoryManager,
    ConfigValidator, ModelCheckpointer, set_random_seeds,
    create_experiment_directory, format_time, calculate_eta
)
from real_time_monitor import RealTimeMonitor, EpisodeState
from performance_analyzer import PerformanceAnalyzer

class ExtendedTrainingManager:
    """최적화 학습 관리자"""

    def __init__(self, experiment_name: str = None, resume_from: str = None):
        self.config = CurriculumConfig()
        self.experiment_name = experiment_name or f"extended_training_{int(time.time())}"
        self.resume_from = resume_from

        # 실험 디렉토리 설정
        self.experiment_dir = create_experiment_directory("experiments", self.experiment_name)
        self.config.SAVE_DIR = os.path.join(self.experiment_dir, "checkpoints")
        self.config.LOG_DIR = os.path.join(self.experiment_dir, "logs")

        # 랜덤 시드 설정
        set_random_seeds(self.config.RANDOM_SEED)

        # 컴포넌트 초기화
        self.env = None
        self.agent = None
        self.logger = None
        self.monitor = None
        self.visualizer = None
        self.checkpointer = None
        self.performance_analyzer = PerformanceAnalyzer()

        # 학습 상태
        self.start_episode = 0
        self.current_stage = 'basic_stabilization'
        self.stage_best_rewards = {}
        self.milestone_achieved = set()

        # 확장 학습 통계
        self.recent_rewards = []
        self.stage_performance = {}
        self.learning_acceleration = 0
        self.convergence_tracker = []

        # 실시간 모니터
        self.real_time_monitor = RealTimeMonitor()
        self.use_real_time_display = True

        # 학습 중단 처리
        self.training_interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

        print(f"Extended Training: {self.experiment_name}")
        print(f"📁 실험 디렉토리: {self.experiment_dir}")

    def _signal_handler(self, signum, frame):
        print("\n⚠️ 학습 중단 신호 수신... 안전하게 종료 중...")
        self.training_interrupted = True
        if hasattr(self, 'real_time_monitor'):
            self.real_time_monitor.cleanup_terminal()

    def initialize_components(self):
        """컴포넌트 초기화"""
        print("🔧 확장 학습 컴포넌트 초기화...")

        # 최신 기법 모니터링 변수들
        self.categorical_metrics = []
        self.curiosity_metrics = []
        self.ucb_metrics = []

        # 환경 초기화
        try:
            self.env = AirSimDroneEnv()
            print("✅ AirSim 환경 연결 성공")
        except Exception as e:
            print(f"❌ AirSim 환경 연결 실패: {e}")
            return False

        # 에이전트 초기화
        self.agent = DQNAgent(self.config)
        print("✅ DQN 에이전트 초기화 완료")

        # 학습 재시작
        if self.resume_from:
            self.start_episode = self.agent.load_model(self.resume_from)
            # 모델 로드 완료

        # 로거 초기화
        self.logger = Logger(self.config.LOG_DIR, self.experiment_name)
        self.logger.log_model_info(self.agent.q_network)

        # 기타 컴포넌트
        self.monitor = PerformanceMonitor()
        self.visualizer = Visualizer(os.path.join(self.experiment_dir, "plots"))
        self.checkpointer = ModelCheckpointer(self.config.SAVE_DIR, max_checkpoints=10)  # 더 많은 체크포인트

        print("✅ 모든 컴포넌트 초기화 완료")
        return True

    def update_stage_configuration(self, episode: int):
        """확장 학습을 위한 동적 설정 업데이트"""
        # 커리큘럼 설정 업데이트
        stage_name, stage_config = self.config.update_config_for_episode(episode)

        # 환경 설정 동적 업데이트
        if hasattr(self.env, 'config'):
            self.env.config.MAX_EPISODE_STEPS = stage_config['max_episode_steps']
            self.env.config.TARGET_RADIUS = stage_config['target_radius']
            self.env.config.REWARDS.update(stage_config['rewards'])

        # 에이전트 설정 동적 업데이트
        new_lr = self.config.get_learning_rate_schedule(episode)
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = new_lr

        # 타겟 업데이트 주기 동적 조정
        new_target_freq = self.config.get_target_update_frequency(episode)
        self.config.TARGET_UPDATE_FREQUENCY = new_target_freq

        # 스테이지 변경 시 최적화 적용
        if stage_name != self.current_stage:
            print(f"\n🎯 스테이지 변경: {self.current_stage} → {stage_name}")
            print(f"📋 새로운 포커스: {stage_config['focus']}")
            print(f"🔄 새로운 타겟 업데이트 주기: {new_target_freq}")

            # 단계별 최적화
            self.optimize_for_stage(stage_name, episode)
            self.current_stage = stage_name

        return stage_name, stage_config

    def optimize_for_stage(self, stage_name: str, episode: int):
        """각 스테이지별 특화 최적화"""
        if stage_name == 'basic_stabilization':
            # 기본 안정화: 충돌 회피 강화
            self.agent.memory.beta = min(1.0, self.agent.memory.beta + 0.05)
        elif stage_name == 'basic_target_detection':
            # 기본 목표 탐지: 탐험-활용 균형 조정
            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = max(self.agent.epsilon * 0.95, 0.3)
        elif stage_name == 'precision_navigation':
            # 정밀 내비게이션: 업데이트 빈도 증가
            self.config.TARGET_UPDATE_FREQUENCY = max(150, self.config.TARGET_UPDATE_FREQUENCY - 50)
        elif stage_name == 'multi_target_handling':
            # 다중 목표: 배치 크기 최적화
            if episode > 600 and self.config.BATCH_SIZE < 64:
                self.config.BATCH_SIZE = min(64, self.config.BATCH_SIZE + 16)
        elif stage_name == 'master_optimization':
            # 마스터 최적화: 미세 조정
            self.agent.memory.beta = 1.0  # 최대 우선순위 적용

        print(f"⚡ {stage_name} 단계용 특화 최적화 적용")

    def run_episode(self, display_episode: int) -> Dict:
        """확장 최적화된 에피소드 실행"""
        episode = display_episode - 1

        # 스테이지 설정 업데이트
        stage_name, stage_config = self.update_stage_configuration(episode)

        # Epsilon 업데이트
        self.agent.epsilon = self.config.get_epsilon_schedule(episode)

        # 에피소드 실행
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0
        targets_found = 0
        efficiency_score = 0
        cumulative_reward = 0
        q_values_history = []

        start_time = time.time()

        while not self.training_interrupted:
            # 현재 상태 정보
            current_pos = self.env.get_current_position()
            target_pos = self.env.config.TARGET_POSITIONS[self.env.current_target_idx]
            distance_to_target = self.env._calculate_distance_to_target()

            # 행동 선택 및 Q값 추적 (타이밍 제어 추가)
            action_start_time = time.time()
            action = self.agent.select_action(state, training=True)

            # 행동 선택 후 최소 대기시간 보장 (안정적인 학습을 위함)
            elapsed_action_time = time.time() - action_start_time
            if elapsed_action_time < self.config.MIN_ACTION_INTERVAL:
                time.sleep(self.config.MIN_ACTION_INTERVAL - elapsed_action_time)

            # Q값 모니터링 (메모리 효율적으로 개선)
            if stage_name == 'master_optimization' and episode_length % 10 == 0:  # 매 10스텝마다만
                with torch.no_grad():
                    state_tensor = self.agent._dict_to_tensor(state)
                    q_values = self.agent.q_network(state_tensor)
                    q_values_history.append(q_values.mean().item())
                    # 텐서 즉시 삭제
                    del state_tensor, q_values

            # 실시간 모니터 업데이트
            if self.use_real_time_display and episode_length % 1 == 0:
                monitor_state = EpisodeState(
                    episode=display_episode,
                    step=episode_length,
                    max_steps=stage_config['max_episode_steps'],
                    reward=episode_reward / max(episode_length, 1),
                    cumulative_reward=cumulative_reward,
                    current_pos=current_pos,
                    target_pos=target_pos,
                    distance_to_target=distance_to_target,
                    targets_found=targets_found,
                    total_targets=len(self.env.config.TARGET_POSITIONS),
                    epsilon=self.agent.epsilon,
                    learning_rate=self.config.get_learning_rate_schedule(episode),
                    stage=stage_name,
                    action=str(action),
                    collision=False,
                    reason="in_progress"
                )
                self.real_time_monitor.step_update(monitor_state)

            # 환경 스텝
            next_state, reward, done, info = self.env.step(action)

            # 고급 보상 추적
            targets_this_episode = info.get('targets_found', 0)
            if targets_this_episode > targets_found:
                efficiency_score += (targets_this_episode - targets_found) * 10

            # 경험 저장
            self.agent.store_experience(state, action, reward, next_state, done)

            # 메모리 효율적인 모델 업데이트
            if len(self.agent.memory) >= self.config.MIN_REPLAY_SIZE:
                # 단계별 업데이트 빈도 조정 (메모리 효율성 고려)
                update_frequency = {
                    'basic_stabilization': 8,       # 매 8스텝마다 (메모리 절약)
                    'basic_target_detection': 6,    # 매 6스텝마다
                    'precision_navigation': 4,      # 매 4스텝마다
                    'multi_target_handling': 3,     # 매 3스텝마다
                    'master_optimization': 2        # 매 2스텝마다
                }.get(stage_name, 4)

                if episode_length % update_frequency == 0:
                    loss = self.agent.update_model()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1

                    # 메모리 정리 (매 업데이트마다)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 상태 업데이트
            state = next_state
            episode_reward += reward
            cumulative_reward += reward
            episode_length += 1
            targets_found = info.get('targets_found', 0)

            # 에피소드 종료 조건
            if done or episode_length >= stage_config['max_episode_steps']:
                break

        # 수렴 추적
        self.convergence_tracker.append(cumulative_reward)
        if len(self.convergence_tracker) > 50:
            self.convergence_tracker.pop(0)

        # 환경 렌더링 정보
        render_info = self.env.render()

        # 최종 실시간 모니터 업데이트
        if self.use_real_time_display:
            final_monitor_state = EpisodeState(
                episode=display_episode,
                step=episode_length,
                max_steps=stage_config['max_episode_steps'],
                reward=episode_reward / max(episode_length, 1),
                cumulative_reward=cumulative_reward,
                current_pos=current_pos,
                target_pos=target_pos,
                distance_to_target=render_info.get('distance_to_target', 0),
                targets_found=targets_found,
                total_targets=len(self.env.config.TARGET_POSITIONS),
                epsilon=self.agent.epsilon,
                learning_rate=self.config.get_learning_rate_schedule(episode),
                stage=stage_name,
                action=str(action),
                collision=info.get('reason') == 'collision',
                reason=info.get('reason', 'unknown')
            )
            self.real_time_monitor.episode_complete(final_monitor_state)

        # 성능 추적 업데이트
        self.update_performance_tracking(cumulative_reward, stage_name)

        # 반환 정보 추출 (render_info 삭제 전에)
        exploration_coverage = render_info.get('exploration_coverage', 0)
        distance_to_target_final = render_info.get('distance_to_target', 0)

        # 에피소드 종료 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 로컬 변수 정리 (큰 객체들)
        del state, render_info
        if q_values_history:
            del q_values_history

        return {
            'reward': cumulative_reward,
            'length': episode_length,
            'loss': episode_loss / max(loss_count, 1),
            'targets_found': targets_found,
            'epsilon': self.agent.epsilon,
            'time': time.time() - start_time,
            'exploration_coverage': exploration_coverage,
            'distance_to_target': distance_to_target_final,
            'reason': info.get('reason', 'unknown'),
            'stage': stage_name,
            'stage_focus': stage_config['focus'],
            'efficiency_score': efficiency_score,
            'learning_rate': self.config.get_learning_rate_schedule(episode),
            'avg_q_value': np.mean(q_values_history) if q_values_history else 0,
            'convergence_trend': np.mean(self.convergence_tracker[-10:]) if len(self.convergence_tracker) >= 10 else 0
        }

    def update_performance_tracking(self, reward: float, stage_name: str):
        """성능 추적 및 수렴 분석"""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:  # 최근 100 에피소드
            self.recent_rewards.pop(0)

        # 스테이지별 성능 기록
        if stage_name not in self.stage_performance:
            self.stage_performance[stage_name] = []
        self.stage_performance[stage_name].append(reward)

        # 학습 가속도 계산
        if len(self.recent_rewards) >= 40:
            recent_avg = np.mean(self.recent_rewards[-20:])
            older_avg = np.mean(self.recent_rewards[-40:-20])
            self.learning_acceleration = recent_avg - older_avg

    def check_milestones(self, episode: int, stats: Dict):
        """확장 마일스톤 달성 확인"""
        if episode in CURRICULUM_MILESTONES and episode not in self.milestone_achieved:
            milestone = CURRICULUM_MILESTONES[episode]
            expected_reward = milestone['expected_reward']

            # 최근 성능 평가 (더 엄격한 기준)
            recent_avg = np.mean(self.recent_rewards) if self.recent_rewards else 0
            achievement_ratio = recent_avg / expected_reward if expected_reward > 0 else 1.0

            # 마일스톤 체크
            print(f"📊 최근 평균 보상: {recent_avg:.2f} (목표: {expected_reward})")
            print(f"🎯 달성도: {achievement_ratio:.1%}")
            print(f"📈 학습 가속도: {self.learning_acceleration:.2f}")

            # 달성 기준 (더 엄격)
            if achievement_ratio >= 0.8 or recent_avg > expected_reward * 0.8:
                print("✅ 마일스톤 달성!")
                self.milestone_achieved.add(episode)
                self.apply_milestone_bonus(episode)
            else:
                print("⚠️ 마일스톤 미달성 - 학습 강화 적용")
                self.apply_learning_boost(episode)

    def apply_milestone_bonus(self, episode: int):
        """마일스톤 달성 시 최적화 보너스"""
        # 타겟 네트워크 즉시 업데이트
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

        # 메모리 최적화
        MemoryManager.optimize_memory()

        # 스케줄러 스텝 (성능 기반)
        recent_avg = np.mean(self.recent_rewards[-20:]) if len(self.recent_rewards) >= 20 else 0
        self.agent.scheduler.step(recent_avg)

        print("🎉 마일스톤 보너스: 타겟 업데이트 + 메모리 최적화 + 스케줄러 조정")

    def apply_learning_boost(self, episode: int):
        """학습 부진 시 적응적 부스트"""
        # 현재 단계에 따른 부스트 전략
        stage_name = self.current_stage

        if stage_name in ['basic_stabilization', 'basic_target_detection']:
            # 초기 단계: 학습률 부스트
            current_lr = self.agent.optimizer.param_groups[0]['lr']
            boost_lr = min(0.002, current_lr * 1.3)
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = boost_lr
            print(f"⚡ 초기 단계 부스트: 학습률 {current_lr:.5f} → {boost_lr:.5f}")

        elif stage_name in ['precision_navigation', 'multi_target_handling']:
            # 중간 단계: 타겟 업데이트 빈도 증가
            self.config.TARGET_UPDATE_FREQUENCY = max(100, self.config.TARGET_UPDATE_FREQUENCY - 50)
            print(f"⚡ 중간 단계 부스트: 타겟 업데이트 주기 → {self.config.TARGET_UPDATE_FREQUENCY}")

        else:
            # 최종 단계: PER 강화
            self.agent.memory.alpha = min(1.0, self.agent.memory.alpha + 0.1)
            print(f"⚡ 최종 단계 부스트: PER Alpha → {self.agent.memory.alpha:.2f}")

    def _log_advanced_metrics(self, episode: int, stats: Dict):
        """최신 기법들의 고급 메트릭 로깅"""
        try:
            # Categorical DQN 메트릭
            if self.config.USE_CATEGORICAL_DQN:
                if hasattr(self.agent.q_network, 'get_categorical_distribution'):
                    # Distribution entropy (exploration 측정)
                    with torch.no_grad():
                        dummy_state = self._get_dummy_state()
                        dist = self.agent.q_network.get_categorical_distribution(dummy_state)
                        entropy = -(dist * torch.log(dist + 1e-8)).sum(dim=-1).mean()
                        self.categorical_metrics.append(entropy.item())

            # UCB exploration 메트릭
            if self.config.USE_UCB_EXPLORATION and hasattr(self.agent, 'action_counts'):
                # Action diversity
                action_probs = self.agent.action_counts / self.agent.total_actions
                action_entropy = -(action_probs * np.log(action_probs + 1e-8)).sum()
                self.ucb_metrics.append(action_entropy)

            # 로그 출력 (간결하게)
            if episode % (self.config.LOG_INTERVAL * 2) == 0:
                # 고급 메트릭 기록

                if self.categorical_metrics:
                    recent_entropy = np.mean(self.categorical_metrics[-10:])
                    print(f"  🎲 분포 엔트로피: {recent_entropy:.3f}")

                if self.ucb_metrics:
                    recent_diversity = np.mean(self.ucb_metrics[-10:])
                    print(f"  🎯 행동 다양성: {recent_diversity:.3f}")

                # GPU memory tracking (silent)
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    # Memory tracked silently

        except Exception as e:
            print(f"⚠️ 고급 메트릭 로깅 오류: {e}")

    def _get_dummy_state(self):
        """더미 상태 생성 (메트릭 계산용)"""
        return {
            'image': torch.randn(1, self.config.IMAGE_CHANNELS,
                               self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH).to(self.agent.device),
            'position': torch.randn(1, 9).to(self.agent.device)
        }

    def train(self):
        """확장 학습 메인 루프"""
        if not self.initialize_components():
            print("❌ 컴포넌트 초기화 실패")
            return

        print(f"Starting Extended Training!")
        print("=" * 70)

        training_start_time = time.time()

        try:
            for episode in range(self.start_episode, self.config.NUM_EPISODES):
                if self.training_interrupted:
                    break

                display_episode = episode + 1

                # 첫 번째 에피소드 시작 구분선
                if display_episode == 1:
                    print("\n" + "=" * 90)
                    # 에피소드 시작
                    print("=" * 90)

                # 에피소드 실행
                episode_stats = self.run_episode(display_episode)

                # 성능 모니터링
                perf_metrics = self.monitor.update()

                # 통합 통계
                combined_stats = {**episode_stats, **perf_metrics}

                # 성능 분석기에 데이터 추가
                combined_stats['collision'] = episode_stats.get('reason') == 'collision'
                self.performance_analyzer.add_episode_data(display_episode, combined_stats)

                # 진행상황 로깅
                if episode % self.config.LOG_INTERVAL == 0:
                    # print_curriculum_progress(episode, self.config)  # 비활성화
                    self.logger.log_episode(display_episode, combined_stats)

                    # 성능 요약
                    recent_avg = np.mean(self.recent_rewards) if self.recent_rewards else 0
                    elapsed_time = time.time() - training_start_time
                    eta = calculate_eta(episode - self.start_episode,
                                      self.config.NUM_EPISODES - self.start_episode,
                                      elapsed_time)

                    print(f"\nEpisode {display_episode:4d}/{self.config.NUM_EPISODES} | "
                          f"보상: {episode_stats['reward']:7.1f} | "
                          f"평균: {recent_avg:7.1f} | "
                          f"Stage: {episode_stats['stage'][:12]:<12} | "
                          f"LR: {episode_stats['learning_rate']:.6f} | "
                          f"ETA: {eta}")

                    # 수렴 분석
                    if len(self.convergence_tracker) >= 30:
                        convergence_std = np.std(self.convergence_tracker[-20:])
                        print(f"[CONVERGENCE] 수렴 안정성: {convergence_std:.2f} | 가속도: {self.learning_acceleration:+.2f}")

                    # 권장사항 출력 (더 적은 빈도)
                    if episode % (self.config.LOG_INTERVAL * 4) == 0:
                        recommendations = get_stage_recommendations(episode, self.config)
                        print(f"[TIPS] 현재 단계 권장사항: {recommendations[0]}")

                # 마일스톤 체크
                self.check_milestones(display_episode, combined_stats)

                # 체크포인트 저장
                if episode % self.config.SAVE_FREQUENCY == 0:
                    self.save_checkpoint(display_episode, combined_stats)

                # 적극적인 메모리 최적화 (RTX 3060Ti 8GB)
                if episode % 50 == 0:
                    MemoryManager.optimize_memory()

                # GPU 메모리 정리 (더 자주 실행)
                if episode % 25 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # GPU 동기화로 완전한 메모리 정리

                # 에피소드 통계 정리 (메모리 누수 방지)
                if hasattr(self, 'recent_rewards') and len(self.recent_rewards) > 200:
                    self.recent_rewards = self.recent_rewards[-100:]  # 최근 100개만 유지

                # 최신 기법 성능 모니터링
                if episode % self.config.LOG_INTERVAL == 0:
                    self._log_advanced_metrics(episode, combined_stats)

        except Exception as e:
            print(f"[ERROR] 학습 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._cleanup(episode if 'episode' in locals() else self.start_episode)

    def save_checkpoint(self, episode: int, stats: Dict):
        """확장 체크포인트 저장"""
        # 스테이지별 최고 성능 모델 저장
        current_stage = stats.get('stage', 'unknown')
        if current_stage not in self.stage_best_rewards:
            self.stage_best_rewards[current_stage] = float('-inf')

        if stats['reward'] > self.stage_best_rewards[current_stage]:
            self.stage_best_rewards[current_stage] = stats['reward']
            stage_best_path = os.path.join(self.config.SAVE_DIR, f"best_{current_stage}.pth")
            self.agent.save_model(stage_best_path, episode)

        # 정기 체크포인트
        self.checkpointer.save_checkpoint(self.agent, episode, stats)

        # 마스터 레벨 달성 시 특별 저장
        if episode >= 950 and stats['reward'] > 250:
            master_path = os.path.join(self.config.SAVE_DIR, f"master_level_ep{episode}.pth")
            self.agent.save_model(master_path, episode)

        # 최종 모델 저장
        if episode >= self.config.NUM_EPISODES - 1:
            final_path = os.path.join(self.config.SAVE_DIR, "complete_model.pth")
            self.agent.save_model(final_path, episode)
            print(f"완성 모델 저장 완료: {final_path}")

    def _cleanup(self, final_episode: int):
        """확장 학습 종료 정리"""
        print("\n" + "=" * 70)
        print("확장 학습 완료!")

        # 성능 분석 요약 출력
        self.performance_analyzer.print_summary()

        # 상세 보고서 생성
        print("\n📈 Generating Extended Performance Analysis Report...")
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            excel_filename = f"DQN_1000EP_Extended_Performance_{timestamp}.xlsx"
            excel_path = self.performance_analyzer.generate_excel_report(excel_filename)

            json_filename = f"DQN_1000EP_Extended_RawData_{timestamp}.json"
            json_path = self.performance_analyzer.save_raw_data(json_filename)

            print(f"✅ 확장 Excel 보고서: {excel_path}")
            print(f"✅ 확장 원시 데이터: {json_path}")

        except Exception as e:
            print(f"❌ 보고서 생성 실패: {e}")

        # 스테이지별 성과 요약
        print("\n📊 스테이지별 최종 성과:")
        for stage, rewards in self.stage_performance.items():
            if rewards:
                avg_reward = np.mean(rewards)
                max_reward = max(rewards)
                improvement = max_reward - rewards[0] if len(rewards) > 1 else 0
                print(f"  {stage}: 평균 {avg_reward:.1f}, 최고 {max_reward:.1f}, 개선 {improvement:+.1f} ({len(rewards)}회)")

        # 마일스톤 달성률
        achievement_rate = len(self.milestone_achieved) / len(CURRICULUM_MILESTONES) * 100
        print(f"\n마일스톤 달성률: {achievement_rate:.1f}% ({len(self.milestone_achieved)}/{len(CURRICULUM_MILESTONES)})")

        # 최종 성능 평가
        final_performance = np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else 0
        readiness_score = min(100, max(0, (final_performance + 200) / 5))

        print(f"\n🎯 최종 성능 점수: {readiness_score:.1f}%")
        if readiness_score >= 85:
            print("✅ 마스터 레벨 달성! 실전 배포 준비 완료!")
        elif readiness_score >= 70:
            print("✅ 고급 수준 달성! 추가 최적화 권장")
        else:
            print("⚠️ 추가 학습이 필요합니다.")

        # 최종 모델 저장
        if hasattr(self, 'agent') and self.agent:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # 최종 완성 모델
                final_model_path = f"models/DQN_1000EP_Final_{timestamp}.pth"
                os.makedirs("models", exist_ok=True)

                model_state = {
                    'q_network_state_dict': self.agent.q_network.state_dict(),
                    'target_network_state_dict': self.agent.target_network.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'config': self.config.__dict__,
                    'final_episode': final_episode,
                    'training_complete': True,
                    'model_type': 'DQN_1000EP_Complete',
                    'final_performance': final_performance,
                    'readiness_score': readiness_score,
                    'milestone_achieved': list(self.milestone_achieved),
                    'save_timestamp': timestamp
                }

                torch.save(model_state, final_model_path)
                print(f"✅ 최종 1000EP 모델 저장: {final_model_path}")

                # 마스터 레벨 추론 모델
                if readiness_score >= 85:
                    master_inference_path = f"models/DQN_Master_Inference_{timestamp}.pth"
                    inference_state = {
                        'q_network_state_dict': self.agent.q_network.state_dict(),
                        'config': {
                            'STATE_SIZE': self.config.STATE_SIZE,
                            'ACTION_SIZE': self.config.ACTION_SIZE,
                            'IMAGE_WIDTH': self.config.IMAGE_WIDTH,
                            'IMAGE_HEIGHT': self.config.IMAGE_HEIGHT,
                            'IMAGE_CHANNELS': self.config.IMAGE_CHANNELS,
                            'CNN_FILTERS': self.config.CNN_FILTERS,
                            'CNN_KERNELS': self.config.CNN_KERNELS,
                            'CNN_STRIDES': self.config.CNN_STRIDES,
                            'HIDDEN_SIZES': self.config.HIDDEN_SIZES,
                            'USE_DUELING_DQN': self.config.USE_DUELING_DQN,
                            'TARGET_POSITIONS': self.config.TARGET_POSITIONS,
                            'TARGET_RADIUS': self.config.TARGET_RADIUS
                        },
                        'model_type': 'DQN_Master_Inference',
                        'training_episodes': final_episode,
                        'performance_score': readiness_score,
                        'save_timestamp': timestamp
                    }

                    torch.save(inference_state, master_inference_path)
                    print(f"✅ 마스터 추론 모델 저장: {master_inference_path}")

            except Exception as e:
                print(f"❌ 최종 모델 저장 실패: {e}")

        # 시각화 생성
        if hasattr(self, 'logger') and self.logger:
            try:
                self.visualizer.plot_training_curves(self.logger.episode_data, "extended_1000_training")
                print("✅ 확장 학습 곡선 저장 완료")
            except Exception as e:
                print(f"⚠️ 시각화 실패: {e}")

        # 실시간 모니터 정리
        if hasattr(self, 'real_time_monitor'):
            self.real_time_monitor.cleanup_terminal()

        # 리소스 정리
        if hasattr(self, 'env') and self.env:
            self.env.close()
        if hasattr(self, 'logger') and self.logger:
            self.logger.close()

        print("확장 학습 완료 - 마스터 레벨 드론 AI 완성!")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='확장 DQN 학습')
    parser.add_argument('--experiment', type=str, default=None, help='실험 이름')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트에서 재시작')
    parser.add_argument('--no-realtime', action='store_true', help='실시간 모니터링 비활성화')

    args = parser.parse_args()

    print("DQN AirSim Extended Training")
    print("Advanced Deep Reinforcement Learning for Master-Level Autonomous Flight")
    print("=" * 70)

    # 커리큘럼 요약
    config = CurriculumConfig()
    print("5단계 확장 커리큘럼:")
    for stage_name, stage_config in config.CURRICULUM_STAGES.items():
        episodes = stage_config['episodes']
        focus = stage_config['focus']
        steps = stage_config['max_episode_steps']
        print(f"  {episodes[0]:3d}-{episodes[1]:3d}: {focus} (Max {steps} steps)")

    print(f"\n{len(CURRICULUM_MILESTONES)}개 마일스톤으로 완전한 마스터 레벨 달성!")
    print("=" * 70)

    # 학습 시작
    trainer = ExtendedTrainingManager(args.experiment, args.resume)

    # 실시간 모니터링 설정
    if args.no_realtime:
        trainer.use_real_time_display = False
        print("실시간 모니터링 비활성화 - 기본 로그 출력")
    else:
        print("실시간 모니터링 활성화 - 고급 동적 콘솔 출력")
        print("학습 중 Ctrl+C로 안전하게 종료 가능")
        time.sleep(2)

    trainer.train()

if __name__ == "__main__":
    main()