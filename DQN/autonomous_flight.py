"""
Autonomous Flight with Trained DQN Model
훈련된 DQN 모델을 사용한 자율 비행
"""

import torch
import numpy as np
import time
import argparse
import os
from typing import Optional, Tuple
import cv2

from config import Config
from environment import AirSimDroneEnv
from network import create_dqn_network
import warnings
warnings.filterwarnings('ignore')


class AutonomousFlightAgent:
    """훈련된 DQN 모델을 사용한 자율 비행 에이전트"""

    def __init__(self, model_path: str, config: Optional[Config] = None):
        """
        Args:
            model_path: 훈련된 모델 파일 경로
            config: 설정 객체 (None이면 모델에서 불러옴)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # 모델 로드
        self.model_data = torch.load(model_path, map_location=self.device)
        print(f"[SUCCESS] Model loaded from: {model_path}")

        # 설정 로드
        if config is None:
            if 'config' in self.model_data:
                # 모델에서 설정 불러오기
                config_dict = self.model_data['config']
                if isinstance(config_dict, dict):
                    # Config 객체 재구성
                    self.config = Config()
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                else:
                    self.config = config_dict
            else:
                self.config = Config()
        else:
            self.config = config

        # 네트워크 생성 및 가중치 로드
        self.q_network = create_dqn_network(self.config)
        self.q_network.load_state_dict(self.model_data['q_network_state_dict'])
        self.q_network.to(self.device)
        self.q_network.eval()  # 추론 모드

        print(f"[SUCCESS] Network initialized and weights loaded")

        # 모델 정보 출력
        if 'model_type' in self.model_data:
            print(f"📊 Model type: {self.model_data['model_type']}")
        if 'training_episodes' in self.model_data:
            print(f"📈 Training episodes: {self.model_data['training_episodes']}")
        if 'save_timestamp' in self.model_data:
            print(f"⏰ Model saved: {self.model_data['save_timestamp']}")

    def select_action(self, state: np.ndarray) -> int:
        """
        상태에서 최적 행동 선택 (추론 모드)

        Args:
            state: 현재 상태

        Returns:
            선택된 행동
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            return action

    def get_action_name(self, action: int) -> str:
        """행동 번호를 이름으로 변환"""
        action_names = {
            0: "HOVER",
            1: "FORWARD",
            2: "BACKWARD",
            3: "LEFT",
            4: "RIGHT",
            5: "UP",
            6: "DOWN"
        }
        return action_names.get(action, f"UNKNOWN_{action}")


class AutonomousFlightController:
    """자율 비행 제어기"""

    def __init__(self, model_path: str, config: Optional[Config] = None):
        """
        Args:
            model_path: 훈련된 모델 파일 경로
            config: 설정 객체
        """
        self.agent = AutonomousFlightAgent(model_path, config)
        self.config = self.agent.config

        # 환경 초기화
        print("🔗 Connecting to AirSim...")
        self.env = AirSimDroneEnv()
        print("[SUCCESS] AirSim environment connected")

        # 통계
        self.flight_stats = {
            'total_steps': 0,
            'targets_found': 0,
            'collisions': 0,
            'start_time': time.time()
        }

    def run_autonomous_flight(self, max_steps: int = 1000, target_mode: str = "single"):
        """
        자율 비행 실행

        Args:
            max_steps: 최대 스텝 수
            target_mode: "single" (단일 목표) 또는 "multi" (다중 목표)
        """
        print(f"\n[INFO] Starting autonomous flight...")
        print(f"🎯 Mode: {target_mode} target")
        print(f"⏱️ Max steps: {max_steps}")
        print("=" * 60)

        # 환경 리셋
        state = self.env.reset()
        done = False
        step = 0

        print(f"🎯 Target position: {self.env.config.TARGET_POSITIONS[self.env.current_target_idx]}")
        print(f"📍 Starting position: {self.env.get_current_position()}")

        try:
            while not done and step < max_steps:
                # 행동 선택
                action = self.agent.select_action(state)
                action_name = self.agent.get_action_name(action)

                # 환경 스텝
                next_state, reward, done, info = self.env.step(action)

                # 통계 업데이트
                self.flight_stats['total_steps'] += 1

                if info.get('reason') == 'target_reached':
                    self.flight_stats['targets_found'] += 1
                    print(f"🎯 Target {self.flight_stats['targets_found']} reached! Reward: {reward:.2f}")

                    if target_mode == "single":
                        print("[SUCCESS] Mission accomplished! Single target reached.")
                        break
                elif info.get('reason') == 'collision':
                    self.flight_stats['collisions'] += 1
                    print(f"[COLLISION] Collision detected! Step: {step}")
                    break

                # 진행 상황 출력 (100스텝마다로 축소)
                if step % 100 == 0:
                    current_pos = self.env.get_current_position()
                    distance = info.get('distance_to_target', 0)
                    print(f"Step {step:4d}: Pos: ({current_pos[0]:6.1f}, {current_pos[1]:6.1f}, {current_pos[2]:6.1f}) | "
                          f"Distance: {distance:6.1f}m | Reward: {reward:6.2f}")

                state = next_state
                step += 1

                # 작은 지연 (시각적 확인용)
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⚠️ Flight interrupted by user")
        except Exception as e:
            print(f"[ERROR] Flight error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.print_flight_summary(step, info.get('reason', 'unknown'))

    def print_flight_summary(self, steps: int, reason: str):
        """비행 요약 출력"""
        flight_time = time.time() - self.flight_stats['start_time']

        print("\n" + "=" * 60)
        print("🏁 AUTONOMOUS FLIGHT SUMMARY")
        print("=" * 60)
        print(f"⏱️ Flight time: {flight_time:.1f}s")
        print(f"👆 Total steps: {steps}")
        print(f"🎯 Targets found: {self.flight_stats['targets_found']}")
        print(f"[STATS] Collisions: {self.flight_stats['collisions']}")
        print(f"🛑 End reason: {reason}")

        # 성능 평가
        if self.flight_stats['targets_found'] > 0:
            efficiency = self.flight_stats['targets_found'] / steps * 100
            print(f"⚡ Efficiency: {efficiency:.2f}% (targets/step)")

        if reason == 'target_reached':
            print("[SUCCESS] MISSION SUCCESSFUL!")
        elif reason == 'collision':
            print("⚠️ Mission ended due to collision")
        else:
            print("⏰ Mission timed out")

        print("=" * 60)

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DQN Autonomous Flight")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pth)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum flight steps (default: 1000)')
    parser.add_argument('--target-mode', type=str, choices=['single', 'multi'],
                       default='single', help='Target mode: single or multi')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous flight mode (multiple missions)')

    args = parser.parse_args()

    # 모델 파일 존재 확인
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    print("[INFO] DQN Autonomous Flight System")
    print("=" * 60)

    # 자율 비행 컨트롤러 초기화
    try:
        controller = AutonomousFlightController(args.model)

        if args.continuous:
            print("🔄 Continuous flight mode activated")
            mission_count = 0
            while True:
                mission_count += 1
                print(f"\n🎯 Starting Mission {mission_count}")
                controller.run_autonomous_flight(args.max_steps, args.target_mode)

                user_input = input("\nContinue to next mission? (y/N): ").lower()
                if user_input != 'y':
                    break
        else:
            controller.run_autonomous_flight(args.max_steps, args.target_mode)

    except Exception as e:
        print(f"[ERROR] Failed to initialize autonomous flight: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.close()

    print("\n[INFO] Autonomous flight session completed")


if __name__ == "__main__":
    main()