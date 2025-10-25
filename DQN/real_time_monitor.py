"""
실시간 학습 모니터링 시스템
이스케이프 문자를 활용한 동적 콘솔 출력
"""

import os
import time
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class EpisodeState:
    """에피소드 현재 상태"""
    episode: int
    step: int
    max_steps: int
    reward: float
    cumulative_reward: float
    current_pos: List[float]
    target_pos: List[float]
    distance_to_target: float
    targets_found: int
    total_targets: int
    epsilon: float
    learning_rate: float
    stage: str
    action: str
    collision: bool
    reason: str

class RealTimeMonitor:
    """실시간 학습 진행 모니터"""

    def __init__(self):
        self.start_time = time.time()
        self.episode_start_time = time.time()
        self.last_update_time = time.time()

        # 터미널 설정
        self.terminal_width = 80
        self.terminal_height = 25

        # 히스토리
        self.reward_history = []
        self.episode_times = []
        self.recent_actions = []

        # 통계
        self.total_episodes = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.worst_reward = float('inf')

        # 헤더 출력 제어
        self.header_displayed = False

        # 색상 코드 (ANSI)
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'reset': '\033[0m',
            'clear_line': '\033[2K',
            'clear_screen': '\033[2J',
            'home': '\033[H'
        }

        # Action symbols (Windows console compatible)
        self.action_emojis = {
            0: "||",  # HOVER
            1: "^^",  # FORWARD
            2: "vv",  # BACKWARD
            3: "<<",  # LEFT
            4: ">>",  # RIGHT
            5: "/\\",  # UP
            6: "\\/"   # DOWN
        }

        self.init_terminal()

    def init_terminal(self):
        """터미널 초기화"""
        # Windows CMD 색상 지원 활성화
        if os.name == 'nt':
            os.system('color')

        # 화면 클리어 및 커서 숨기기
        print(self.colors['clear_screen'] + self.colors['home'], end='')
        print('\033[?25l', end='')  # 커서 숨기기
        sys.stdout.flush()

    def cleanup_terminal(self):
        """터미널 정리"""
        print('\033[?25h', end='')  # 커서 보이기
        print(self.colors['reset'])
        sys.stdout.flush()

    def get_color_by_value(self, value: float, thresholds: Dict[str, float]) -> str:
        """값에 따른 색상 반환"""
        if value >= thresholds.get('excellent', 100):
            return self.colors['green']
        elif value >= thresholds.get('good', 50):
            return self.colors['cyan']
        elif value >= thresholds.get('average', 0):
            return self.colors['yellow']
        else:
            return self.colors['red']

    def format_time(self, seconds: float) -> str:
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

    def create_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """ASCII progress bar creation"""
        if total == 0:
            return "=" * width

        progress = current / total
        filled = int(progress * width)
        bar = "=" * filled + "-" * (width - filled)
        percentage = progress * 100

        color = self.get_color_by_value(percentage, {'excellent': 80, 'good': 50, 'average': 20})
        return f"{color}[{bar}]{self.colors['reset']} {percentage:5.1f}%"


    def draw_header(self) -> List[str]:
        """헤더 그리기"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time

        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.colors['bold']}{self.colors['cyan']}DQN AirSim Real-time Training Monitor{self.colors['reset']}")
        lines.append(f"Start Time: {current_time} | Elapsed: {self.format_time(elapsed)}")
        lines.append("=" * 80)

        return lines

    def draw_episode_info(self, state: EpisodeState) -> List[str]:
        """에피소드 정보 그리기"""
        lines = []

        # 에피소드 기본 정보
        episode_progress = self.create_progress_bar(state.step, state.max_steps, 30)
        episode_time = time.time() - self.episode_start_time

        target_coords = f"({state.target_pos[0]:.1f}, {state.target_pos[1]:.1f}, {state.target_pos[2]:.1f})"
        lines.append(f"{self.colors['bold']}[EP] EPISODE {state.episode:3d}{self.colors['reset']} | Target: {self.colors['cyan']}{target_coords}{self.colors['reset']}")
        lines.append(f"Steps: {state.step:3d}/{state.max_steps:3d} {episode_progress}")
        lines.append(f"Time: {self.format_time(episode_time)} | Stage: {self.colors['magenta']}{state.stage}{self.colors['reset']}")

        return lines

    def draw_position_info(self, state: EpisodeState) -> List[str]:
        """Draw position information"""
        lines = []

        # Current position
        pos_str = f"({state.current_pos[0]:6.1f}, {state.current_pos[1]:6.1f}, {state.current_pos[2]:6.1f})"
        target_str = f"({state.target_pos[0]:6.1f}, {state.target_pos[1]:6.1f}, {state.target_pos[2]:6.1f})"

        # Color based on distance
        distance_color = self.get_color_by_value(
            100 - state.distance_to_target,
            {'excellent': 95, 'good': 80, 'average': 60}
        )

        lines.append(f"{self.colors['bold']}[POS] Position Info{self.colors['reset']}")
        lines.append(f"Current Position: {self.colors['white']}{pos_str}{self.colors['reset']}")
        lines.append(f"Target Position: {self.colors['green']}{target_str}{self.colors['reset']}")
        lines.append(f"Distance: {distance_color}{state.distance_to_target:6.1f}m{self.colors['reset']} | "
                    f"Targets Found: {self.colors['cyan']}{state.targets_found}/{state.total_targets}{self.colors['reset']}")

        return lines

    def draw_reward_info(self, state: EpisodeState) -> List[str]:
        """Draw reward information"""
        lines = []

        # Reward colors
        reward_color = self.get_color_by_value(
            state.reward,
            {'excellent': 50, 'good': 10, 'average': -10}
        )

        cumulative_color = self.get_color_by_value(
            state.cumulative_reward,
            {'excellent': 100, 'good': 50, 'average': 0}
        )

        lines.append(f"{self.colors['bold']}[$] Reward Info{self.colors['reset']}")
        lines.append(f"Current Reward: {reward_color}{state.reward:8.2f}{self.colors['reset']}")
        lines.append(f"Cumulative Reward: {cumulative_color}{state.cumulative_reward:8.2f}{self.colors['reset']}")

        # Calculate average if reward history exists
        if self.reward_history:
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            avg_color = self.get_color_by_value(avg_reward, {'excellent': 50, 'good': 10, 'average': -10})
            lines.append(f"Average Reward: {avg_color}{avg_reward:8.2f}{self.colors['reset']} | "
                        f"Best: {self.colors['green']}{self.best_reward:6.1f}{self.colors['reset']} | "
                        f"Worst: {self.colors['red']}{self.worst_reward:6.1f}{self.colors['reset']}")

        return lines

    def draw_learning_info(self, state: EpisodeState) -> List[str]:
        """Draw learning information"""
        lines = []

        # Epsilon bar
        epsilon_bar = self.create_progress_bar(int(state.epsilon * 100), 100, 15)

        lines.append(f"{self.colors['bold']}[AI] Learning Info{self.colors['reset']}")
        lines.append(f"Epsilon: {epsilon_bar}")
        lines.append(f"Learning Rate: {self.colors['white']}{state.learning_rate:.6f}{self.colors['reset']}")

        return lines



    def draw_status_line(self, state: EpisodeState) -> List[str]:
        """Draw status line"""
        lines = []

        # Termination reason
        reason_color = self.colors['green'] if state.reason == 'target_reached' else \
                      self.colors['red'] if state.reason == 'collision' else \
                      self.colors['yellow']

        lines.append("─" * 80)
        lines.append(f"Status: {reason_color}{state.reason}{self.colors['reset']} | "
                    f"FPS: {1.0 / max(0.001, time.time() - self.last_update_time):4.1f} | "
                    f"Updated: {datetime.now().strftime('%H:%M:%S')}")

        return lines

    def update_display(self, state: EpisodeState):
        """전체 화면 업데이트"""
        # 첫 번째 업데이트시에만 헤더 출력
        if not self.header_displayed:
            print(self.colors['clear_screen'] + self.colors['home'], end='')
            header_lines = self.draw_header()
            for line in header_lines:
                print(f"{self.colors['clear_line']}{line:<79}")
            self.header_displayed = True
            print()  # 헤더 후 빈 줄

        # 커서를 헤더 이후 위치로 이동 (헤더는 4줄)
        print('\033[5;1H', end='')  # 5번째 줄, 1번째 열로 이동

        # 모든 라인들 수집 (헤더 제외)
        all_lines = []
        all_lines.extend(self.draw_episode_info(state))
        all_lines.append("")
        all_lines.extend(self.draw_position_info(state))
        all_lines.append("")
        all_lines.extend(self.draw_reward_info(state))
        all_lines.append("")
        all_lines.extend(self.draw_learning_info(state))
        all_lines.append("")
        all_lines.extend(self.draw_status_line(state))

        # 화면에 출력
        for line in all_lines:
            print(f"{self.colors['clear_line']}{line:<79}")

        # 버퍼 플러시
        sys.stdout.flush()
        self.last_update_time = time.time()

    def update_statistics(self, state: EpisodeState):
        """통계 업데이트"""
        # 에피소드 변경 감지 (에피소드 번호가 변경되었을 때만)
        if not hasattr(self, 'last_episode') or self.last_episode != state.episode:
            if hasattr(self, 'last_episode') and self.reward_history:  # 이전 에피소드가 있었다면
                episode_time = time.time() - self.episode_start_time
                self.episode_times.append(episode_time)

            # 새 에피소드 시작 구분선 출력
            if hasattr(self, 'last_episode'):  # 첫 번째 에피소드가 아닌 경우에만
                print("\n" + "=" * 80)
                print(f"[>>] Starting Episode {state.episode}")
                print("=" * 80)

            self.episode_start_time = time.time()
            self.total_episodes = state.episode  # 현재 에피소드 번호 사용
            self.last_episode = state.episode

        # 보상 히스토리 업데이트
        if state.cumulative_reward != 0:
            if not self.reward_history or self.reward_history[-1] != state.cumulative_reward:
                self.reward_history.append(state.cumulative_reward)
                if len(self.reward_history) > 50:  # 최근 50개만 유지
                    self.reward_history.pop(0)

                # 최고/최저 보상 업데이트
                self.best_reward = max(self.best_reward, state.cumulative_reward)
                self.worst_reward = min(self.worst_reward, state.cumulative_reward)

        # 행동 히스토리 업데이트
        if state.action.isdigit():
            action_int = int(state.action)
            if not self.recent_actions or self.recent_actions[-1] != action_int:
                self.recent_actions.append(action_int)
                if len(self.recent_actions) > 20:  # 최근 20개만 유지
                    self.recent_actions.pop(0)

        self.total_steps += 1

    def step_update(self, state: EpisodeState):
        """스텝별 업데이트 (매 스텝마다 호출)"""
        self.update_statistics(state)
        self.update_display(state)

    def episode_complete(self, state: EpisodeState):
        """에피소드 완료 시 호출"""
        # 최종 상태 업데이트
        self.update_statistics(state)

        # 에피소드 완료 정보를 터미널에 통합 표시
        self.display_episode_completion(state)

    def display_episode_completion(self, state: EpisodeState):
        """에피소드 완료 정보를 터미널에 통합 표시"""
        # 화면 클리어 후 완료 정보 표시
        print(f"\033[2J\033[H")  # Clear screen and move cursor to top

        # 헤더 표시
        header_lines = self.draw_header()
        for line in header_lines:
            print(f"{line:<79}")

        print("")

        # 에피소드 완료 상태
        completion_lines = self.draw_episode_completion_info(state)
        for line in completion_lines:
            print(f"{line:<79}")

        print("")

        # 최종 상태 정보
        final_lines = []
        final_lines.extend(self.draw_position_info(state))
        final_lines.append("")
        final_lines.extend(self.draw_reward_info(state))
        final_lines.append("")
        final_lines.extend(self.draw_learning_info(state))

        for line in final_lines:
            print(f"{line:<79}")

        print("")

        # 에피소드 완료 구분선
        print("=" * 80)
        print()

        # 버퍼 플러시
        sys.stdout.flush()

    def draw_episode_completion_info(self, state: EpisodeState) -> List[str]:
        """에피소드 완료 정보 그리기"""
        lines = []

        # 에피소드 상태에 따른 색상 및 아이콘
        status_info = self.get_episode_status_info(state.reason)
        episode_time = time.time() - self.episode_start_time

        lines.append(f"{self.colors['bold']}[DONE] EPISODE {state.episode} COMPLETED{self.colors['reset']}")
        lines.append("")
        lines.append(f"{status_info['icon']} {self.colors['bold']}{status_info['color']}{status_info['message']}{self.colors['reset']}")
        lines.append("")

        # 완료 정보 요약
        lines.append(f"{self.colors['bold']}[SUM] Episode Summary{self.colors['reset']}")
        lines.append(f"[T] Duration: {self.format_time(episode_time)}")
        lines.append(f"[#] Steps: {state.step}")
        lines.append(f"[$] Final Reward: {state.cumulative_reward:.2f}")
        lines.append(f"[T] Targets Found: {state.targets_found}/{state.total_targets}")
        lines.append(f"[E] Final Epsilon: {state.epsilon:.3f}")

        # 성능 평가
        if state.step > 0:
            efficiency = (state.targets_found / state.step) * 100
            lines.append(f"[%] Efficiency: {efficiency:.2f}%")

        return lines

    def get_episode_status_info(self, reason: str) -> dict:
        """에피소드 종료 이유에 따른 상태 정보"""
        status_map = {
            'target_reached': {
                'icon': '[T]',
                'color': self.colors['green'],
                'message': 'SUCCESS - Target Reached!'
            },
            'all_targets_found': {
                'icon': '[**]',
                'color': self.colors['green'],
                'message': 'EXCELLENT - All Targets Found!'
            },
            'collision': {
                'icon': '[X]',
                'color': self.colors['red'],
                'message': 'COLLISION - Episode Terminated'
            },
            'out_of_bounds': {
                'icon': '[!]',
                'color': self.colors['red'],
                'message': 'OUT OF BOUNDS - Boundary Exceeded'
            },
            'max_steps': {
                'icon': '[T]',
                'color': self.colors['yellow'],
                'message': 'TIMEOUT - Maximum Steps Reached'
            },
            'timeout': {
                'icon': '[T]',
                'color': self.colors['yellow'],
                'message': 'TIMEOUT - Time Limit Exceeded'
            }
        }

        return status_map.get(reason, {
            'icon': '[?]',
            'color': self.colors['blue'],
            'message': f'UNKNOWN - {reason}'
        })

    def get_episode_status_display(self, reason: str) -> str:
        """에피소드 종료 이유에 따른 상태 표시"""
        status_map = {
            'target_reached': f"{self.colors['bold']}{self.colors['green']}[OK] SUCCESS - Target Reached!{self.colors['reset']}",
            'all_targets_found': f"{self.colors['bold']}{self.colors['green']}[**] SUCCESS - All Targets Found!{self.colors['reset']}",
            'collision': f"{self.colors['bold']}{self.colors['red']}[X] COLLISION - Crash Occurred!{self.colors['reset']}",
            'out_of_bounds': f"{self.colors['bold']}{self.colors['red']}[!] OUT OF BOUNDS - Boundary Exceeded!{self.colors['reset']}",
            'max_steps': f"{self.colors['bold']}{self.colors['yellow']}[T] TIMEOUT - Time Exceeded!{self.colors['reset']}",
            'timeout': f"{self.colors['bold']}{self.colors['yellow']}[T] TIMEOUT - Time Exceeded!{self.colors['reset']}",
        }

        return status_map.get(reason, f"{self.colors['bold']}{self.colors['blue']}❓ UNKNOWN - {reason}{self.colors['reset']}")

# 실시간 모니터 사용 예제
def create_sample_state(episode: int, step: int) -> EpisodeState:
    """샘플 상태 생성 (테스트용)"""
    import random

    return EpisodeState(
        episode=episode,
        step=step,
        max_steps=400,
        reward=random.uniform(-10, 20),
        cumulative_reward=random.uniform(-50, 150),
        current_pos=[random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-15, -5)],
        target_pos=[15.0, 25.0, -10.0],
        distance_to_target=random.uniform(5, 50),
        targets_found=random.randint(0, 2),
        total_targets=4,
        epsilon=max(0.1, 1.0 - episode * 0.002),
        learning_rate=0.0005,
        stage="precision_navigation",
        action=str(random.randint(0, 6)),
        collision=random.random() < 0.1,
        reason=random.choice(["in_progress", "target_reached", "collision", "timeout"])
    )

if __name__ == "__main__":
    # 테스트 실행
    monitor = RealTimeMonitor()

    try:
        print("실시간 모니터 테스트 시작... (Ctrl+C로 종료)")

        for episode in range(1, 6):
            for step in range(1, 101):
                state = create_sample_state(episode, step)
                monitor.step_update(state)
                time.sleep(0.05)  # 0.05초 간격으로 업데이트

            # 에피소드 완료
            final_state = create_sample_state(episode, 100)
            final_state.reason = "target_reached"
            monitor.episode_complete(final_state)
            time.sleep(2)  # 2초 대기

    except KeyboardInterrupt:
        print(f"\n{monitor.colors['yellow']}테스트 중단됨{monitor.colors['reset']}")
    finally:
        monitor.cleanup_terminal()
        print("모니터 테스트 완료!")