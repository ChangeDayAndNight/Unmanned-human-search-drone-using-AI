"""
Advanced DQN Agent with latest techniques:
- Double DQN
- Dueling DQN
- Prioritized Experience Replay (PER)
- Multi-step Learning
- Mixed Precision Training
- Gradient Clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import pickle
import os
from dataclasses import dataclass

from config import Config
from network import create_dqn_network, DQNNetwork

# Experience 정의
Experience = namedtuple('Experience',
                       ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

@dataclass
class TrainingStats:
    """학습 통계"""
    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    loss: float = 0.0
    q_value_mean: float = 0.0
    epsilon: float = 0.0
    targets_found: int = 0
    exploration_coverage: float = 0.0

class SumTree:
    """Prioritized Experience Replay를 위한 Sum Tree 자료구조"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """우선순위 변경을 트리 상위로 전파"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """주어진 값에 해당하는 리프 노드 찾기"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """총 우선순위 합"""
        return self.tree[0]

    def add(self, priority: float, data: Experience):
        """새로운 경험 추가"""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """우선순위 업데이트"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Experience]:
        """우선순위에 따른 샘플링"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def add(self, state: Dict, action: int, reward: float,
            next_state: Dict, done: bool):
        """경험 추가"""
        experience = Experience(state, action, reward, next_state, done, self.max_priority)
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """우선순위 기반 샘플링"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)

            batch.append(experience)
            idxs.append(idx)
            priorities.append(priority)

        # Importance Sampling weights 계산
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, np.array(idxs), is_weights

    def update_priorities(self, idxs: np.ndarray, errors: np.ndarray):
        """TD 오차를 바탕으로 우선순위 업데이트"""
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

class DQNAgent:
    """Advanced DQN Agent"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        # 네트워크 초기화
        self.q_network = create_dqn_network(config, "standard").to(self.device)
        self.target_network = create_dqn_network(config, "standard").to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 옵티마이저 (AdamW with weight decay) - 최적화됨
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5,  # 가중치 감쇠 최적화
            amsgrad=True,
            eps=1e-8
        )

        # 학습률 스케줄러 - 최적화됨
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.9, patience=150
        )

        # 경험 재생 버퍼
        self.memory = PrioritizedReplayBuffer(
            config.REPLAY_BUFFER_SIZE,
            config.PER_ALPHA,
            config.PER_BETA_START
        )

        # Exploration 파라미터
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_END

        # Multi-step learning
        self.multi_step_buffer = deque(maxlen=config.MULTI_STEP_N)
        self.n_step = config.MULTI_STEP_N

        # 학습 통계
        self.stats = TrainingStats()
        self.update_count = 0

        # Mixed Precision Training
        if config.MIXED_PRECISION:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # 성능 모니터링 (메모리 최적화)
        self.q_values_history = deque(maxlen=100)  # 크기 축소
        self.loss_history = deque(maxlen=100)  # 크기 축소

        print(f"DQN Agent 초기화 완료 - Device: {self.device}")
        print(f"네트워크 파라미터 수: {sum(p.numel() for p in self.q_network.parameters()):,}")

    def select_action(self, state: Dict, training: bool = True) -> int:
        """행동 선택 (행동 다양성 보장)"""
        with torch.no_grad():
            state_tensor = self._dict_to_tensor(state)

            # Noisy Networks 활성화 (내장 exploration)
            if self.config.USE_NOISY_NETS and training:
                self.q_network.reset_noise()

            q_values = self.q_network(state_tensor)
            self.q_values_history.append(q_values.mean().item())

            # UCB-based exploration (비활성화됨)
            if training and self.config.USE_UCB_EXPLORATION:
                return self._ucb_action_selection(q_values.squeeze(0))

            # 강화된 epsilon-greedy (Noisy Networks와 병행)
            if training and random.random() < self.epsilon:
                # 균등한 탐험을 통한 더 나은 Q-value 학습
                selected_action = np.random.choice(self.config.ACTION_SIZE)
                return selected_action

            # Greedy action selection
            selected_action = q_values.argmax().item()
            return selected_action

    def _ucb_action_selection(self, q_values: torch.Tensor) -> int:
        """Upper Confidence Bound 기반 행동 선택"""
        # Action counts tracking
        if not hasattr(self, 'action_counts'):
            self.action_counts = np.ones(self.config.ACTION_SIZE)  # Initialize with 1 to avoid division by zero
            self.total_actions = self.config.ACTION_SIZE

        # UCB calculation
        exploration_bonus = self.config.UCB_C * np.sqrt(np.log(self.total_actions) / self.action_counts)
        ucb_values = q_values.cpu().numpy() + exploration_bonus

        # Select action with highest UCB value
        action = np.argmax(ucb_values)

        # Update counts
        self.action_counts[action] += 1
        self.total_actions += 1

        return action

    def store_experience(self, state: Dict, action: int, reward: float,
                        next_state: Dict, done: bool):
        """경험 저장 (Multi-step 지원)"""
        self.multi_step_buffer.append((state, action, reward, next_state, done))

        if len(self.multi_step_buffer) == self.n_step:
            # Multi-step return 계산
            multi_step_reward = 0
            gamma = self.config.GAMMA

            for i, (_, _, r, _, d) in enumerate(self.multi_step_buffer):
                multi_step_reward += (gamma ** i) * r
                if d:
                    break

            # 첫 번째 state-action과 마지막 next_state 사용
            first_transition = self.multi_step_buffer[0]
            last_transition = self.multi_step_buffer[-1]

            self.memory.add(
                first_transition[0],  # state
                first_transition[1],  # action
                multi_step_reward,    # multi-step reward
                last_transition[3],   # next_state
                last_transition[4]    # done
            )

    def update_model(self) -> float:
        """모델 업데이트 (Categorical DQN과 Curiosity 지원)"""
        if len(self.memory) < self.config.MIN_REPLAY_SIZE:
            return 0.0

        # 배치 샘플링
        batch, idxs, is_weights = self.memory.sample(self.config.BATCH_SIZE)

        # 텐서 변환
        states = self._batch_to_tensor([exp.state for exp in batch])
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states = self._batch_to_tensor([exp.next_state for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool).to(self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # Update dropout rate based on training progress
        if hasattr(self.q_network, 'update_dropout_rate'):
            progress = self.update_count / 100000  # Normalize progress
            self.q_network.update_dropout_rate(progress)

        with torch.cuda.amp.autocast(enabled=self.config.MIXED_PRECISION):
            # Categorical vs Standard DQN
            if self.config.USE_CATEGORICAL_DQN:
                dqn_loss, td_errors = self._categorical_loss(states, actions, rewards, next_states, dones, is_weights)
            else:
                dqn_loss, td_errors = self._standard_loss(states, actions, rewards, next_states, dones, is_weights)

            # Add curiosity loss
            curiosity_loss = torch.tensor(0.0, device=self.device)
            if self.config.USE_CURIOSITY and hasattr(self.q_network, 'compute_curiosity_loss'):
                curiosity_loss = self.q_network.compute_curiosity_loss(states, next_states, actions)

            # Total loss
            total_loss = dqn_loss + self.config.CURIOSITY_BETA * curiosity_loss

        # 역전파
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()

        # PER 우선순위 업데이트 (메모리 최적화)
        with torch.no_grad():
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors_np)

        # 텐서 메모리 정리
        del states, actions, rewards, next_states, dones, is_weights, td_errors
        if 'current_q_values' in locals():
            del current_q_values

        # Beta 어닐링
        self.memory.beta = min(1.0, self.memory.beta +
                              (1.0 - self.config.PER_BETA_START) / 100000)

        # Target network 업데이트
        self.update_count += 1
        if self.update_count % self.config.TARGET_UPDATE_FREQUENCY == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 통계 업데이트
        self.loss_history.append(total_loss.item())

        return total_loss.item()

    def _standard_loss(self, states, actions, rewards, next_states, dones, is_weights):
        """Standard DQN loss calculation"""
        # 현재 Q 값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q 값 계산 (Double DQN)
        with torch.no_grad():
            if self.config.USE_DOUBLE_DQN:
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (self.config.GAMMA ** self.n_step) * next_q_values * ~dones

        # TD 오차 계산
        td_errors = current_q_values - target_q_values

        # Importance Sampling으로 가중치 적용
        loss = (is_weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

        return loss, td_errors

    def _categorical_loss(self, states, actions, rewards, next_states, dones, is_weights):
        """Categorical DQN loss calculation"""
        batch_size = states['image'].size(0)

        # Current distribution
        current_dist = self.q_network.get_categorical_distribution(states)
        current_dist = current_dist[range(batch_size), actions]

        # Target distribution
        with torch.no_grad():
            # Double DQN for action selection
            if self.config.USE_DOUBLE_DQN:
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(1)
            else:
                next_q_values = self.target_network(next_states)
                next_actions = next_q_values.argmax(1)

            next_dist = self.target_network.get_categorical_distribution(next_states)
            next_dist = next_dist[range(batch_size), next_actions]

            # Categorical projection
            target_dist = self._categorical_projection(rewards, next_dist, dones)

        # Cross-entropy loss
        log_current_dist = torch.log(current_dist + 1e-8)
        loss_per_sample = -(target_dist * log_current_dist).sum(1)
        loss = (is_weights * loss_per_sample).mean()

        # TD errors (for PER priority update)
        current_q = torch.sum(current_dist * self.q_network.support, dim=1)
        target_q = torch.sum(target_dist * self.q_network.support, dim=1)
        td_errors = current_q - target_q

        return loss, td_errors

    def _categorical_projection(self, rewards, next_dist, dones):
        """Project categorical distribution for Bellman update"""
        batch_size = rewards.size(0)
        delta_z = self.q_network.delta_z
        v_min, v_max = self.q_network.v_min, self.q_network.v_max
        num_atoms = self.q_network.num_atoms

        # Compute Tz (Bellman operator)
        Tz = rewards.unsqueeze(1) + (~dones).unsqueeze(1).float() * (self.config.GAMMA ** self.n_step) * self.q_network.support.unsqueeze(0)
        Tz = Tz.clamp(min=v_min, max=v_max)

        # Compute projection indices
        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Distribute probability mass
        target_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=self.device).long().unsqueeze(1).expand(batch_size, num_atoms)

        target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return target_dist

    def update_epsilon(self):
        """Epsilon 감소"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str, episode: int = None):
        """모델 저장"""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'stats': self.stats,
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'episode': episode
        }

        if self.scaler:
            save_dict['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(save_dict, filepath)
        print(f"모델 저장 완료: {filepath}")

    def load_model(self, filepath: str) -> int:
        """모델 로드"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 존재하지 않습니다: {filepath}")
            return 0

        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.stats = checkpoint.get('stats', TrainingStats())
        self.epsilon = checkpoint.get('epsilon', self.config.EPSILON_START)
        self.update_count = checkpoint.get('update_count', 0)

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        episode = checkpoint.get('episode', 0)
        print(f"모델 로드 완료: {filepath}, Episode: {episode}")
        return episode

    def _dict_to_tensor(self, state: Dict) -> Dict:
        """상태 딕셔너리를 텐서로 변환"""
        return {
            'image': torch.FloatTensor(state['image']).unsqueeze(0).to(self.device),
            'position': torch.FloatTensor(state['position']).unsqueeze(0).to(self.device)
        }

    def _batch_to_tensor(self, states: List[Dict]) -> Dict:
        """배치 상태를 텐서로 변환 (메모리 효율적 개선)"""
        # 메모리 효율적인 스택 연산
        images = np.stack([state['image'] for state in states])
        positions = np.stack([state['position'] for state in states])

        # 직접 GPU로 전송하여 중간 CPU 텐서 생성 방지
        result = {
            'image': torch.from_numpy(images).float().to(self.device, non_blocking=True),
            'position': torch.from_numpy(positions).float().to(self.device, non_blocking=True)
        }

        # 중간 numpy 배열 메모리 해제
        del images, positions

        return result

    def get_stats(self) -> Dict:
        """현재 통계 반환"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'update_count': self.update_count,
            'avg_q_value': np.mean(list(self.q_values_history)) if self.q_values_history else 0,
            'avg_loss': np.mean(list(self.loss_history)) if self.loss_history else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def set_training_mode(self, training: bool):
        """훈련/평가 모드 설정"""
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()

if __name__ == "__main__":
    # 에이전트 테스트
    config = Config()
    config.validate_config()
    config.create_directories()

    print("DQN Agent 테스트 시작...")

    agent = DQNAgent(config)

    # 더미 상태 생성
    dummy_state = {
        'image': np.random.rand(config.IMAGE_CHANNELS, config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        'position': np.random.rand(config.POSITION_DIM)
    }

    # 행동 선택 테스트
    action = agent.select_action(dummy_state)

    # 경험 저장 테스트
    next_state = {
        'image': np.random.rand(config.IMAGE_CHANNELS, config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        'position': np.random.rand(config.POSITION_DIM)
    }

    agent.store_experience(dummy_state, action, 1.0, next_state, False)
    print(f"메모리 크기: {len(agent.memory)}")

    # 통계 출력
    stats = agent.get_stats()
    print("현재 통계:", stats)

    print("Agent 테스트 완료!")