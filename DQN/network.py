"""
Advanced DQN Network Architecture
Double DQN + Dueling DQN + Multi-step + Mixed Precision 지원
RTX 3060Ti 8GB VRAM 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config import Config

class NoisyLinear(nn.Module):
    """Noisy Networks for Exploration (Fortunato et al., 2017)"""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 학습 가능한 파라미터
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 노이즈용 버퍼 (학습되지 않음)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """파라미터 초기화"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """팩토라이즈된 가우시안 노이즈 생성"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """노이즈 리셋"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

class ConvBlock(nn.Module):
    """최적화된 CNN 블록 (Convolution + ReLU, BatchNorm 제거로 메모리 절약)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x

class AttentionModule(nn.Module):
    """Self-Attention 모듈 (최신 기법)"""

    def __init__(self, features: int):
        super(AttentionModule, self).__init__()
        self.features = features
        self.attention = nn.MultiheadAttention(features, num_heads=4, batch_first=True)  # 헤드 수 최적화
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x.squeeze(1)  # (batch, features)

class CuriosityModule(nn.Module):
    """Intrinsic Curiosity Module (ICM) for exploration"""

    def __init__(self, feature_size: int, action_size: int):
        super(CuriosityModule, self).__init__()

        # Inverse model: predicts action from state features
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_size)
        )

        # Forward model: predicts next state features from current state and action
        self.forward_net = nn.Sequential(
            nn.Linear(feature_size + action_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_size)
        )

    def forward(self, current_features, next_features, actions):
        # Inverse model
        concat_features = torch.cat([current_features, next_features], dim=1)
        pred_actions = self.inverse_net(concat_features)

        # Forward model
        action_onehot = F.one_hot(actions, num_classes=self.forward_net[0].in_features - current_features.shape[1]).float()
        concat_state_action = torch.cat([current_features, action_onehot], dim=1)
        pred_next_features = self.forward_net(concat_state_action)

        return pred_actions, pred_next_features

class DQNNetwork(nn.Module):
    """
    Advanced DQN Network with:
    - Dueling Architecture
    - Noisy Networks (optional)
    - Categorical DQN (distributional RL)
    - Attention Mechanism
    - Curiosity Module
    - Dynamic Dropout
    """

    def __init__(self, config: Config, use_noisy: bool = None):
        super(DQNNetwork, self).__init__()
        self.config = config
        self.use_noisy = use_noisy if use_noisy is not None else config.USE_NOISY_NETS
        self.use_categorical = config.USE_CATEGORICAL_DQN

        # Categorical DQN parameters
        if self.use_categorical:
            self.num_atoms = config.CATEGORICAL_ATOMS
            self.v_min = config.V_MIN
            self.v_max = config.V_MAX
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.num_atoms))

        # CNN for image processing
        self.cnn = self._build_cnn()

        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size()

        # Position encoding (최적화됨)
        self.position_encoder = nn.Sequential(
            nn.Linear(config.POSITION_DIM, 64),   # 설정에서 가져오도록 수정
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )

        # Feature fusion (수정됨 - 동적 드롭아웃)
        combined_features = self.cnn_output_size + 128  # 위치 인코더 출력 크기 변경에 따른 수정

        # Dynamic dropout
        if config.USE_DROPOUT_SCHEDULING:
            self.dropout = nn.Dropout(config.INITIAL_DROPOUT)
        else:
            self.dropout = nn.Dropout(0.2)

        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_features, config.HIDDEN_SIZES[0]),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(config.HIDDEN_SIZES[0], config.HIDDEN_SIZES[1]),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.attention = AttentionModule(config.HIDDEN_SIZES[1])

        # Curiosity module
        if config.USE_CURIOSITY:
            self.curiosity = CuriosityModule(config.HIDDEN_SIZES[1], config.ACTION_SIZE)

        # Dueling DQN head
        if config.USE_DUELING_DQN:
            if self.use_categorical:
                self.value_head = self._build_head(config.HIDDEN_SIZES[1], self.num_atoms)
                self.advantage_head = self._build_head(config.HIDDEN_SIZES[1], config.ACTION_SIZE * self.num_atoms)
            else:
                self.value_head = self._build_head(config.HIDDEN_SIZES[1], 1)
                self.advantage_head = self._build_head(config.HIDDEN_SIZES[1], config.ACTION_SIZE)
        else:
            if self.use_categorical:
                self.q_head = self._build_head(config.HIDDEN_SIZES[1], config.ACTION_SIZE * self.num_atoms)
            else:
                self.q_head = self._build_head(config.HIDDEN_SIZES[1], config.ACTION_SIZE)

        # 가중치 초기화
        self.apply(self._init_weights)

    def _build_cnn(self) -> nn.Module:
        """CNN 네트워크 구성"""
        layers = []
        in_channels = self.config.IMAGE_CHANNELS

        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(self.config.CNN_FILTERS, self.config.CNN_KERNELS, self.config.CNN_STRIDES)
        ):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((4, 4)))  # 고정 크기 출력
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _get_cnn_output_size(self) -> int:
        """CNN 출력 크기 계산"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.config.IMAGE_CHANNELS,
                                    self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)
            output = self.cnn(dummy_input)
            return output.shape[1]

    def _build_head(self, input_size: int, output_size: int) -> nn.Module:
        """DQN 헤드 구성 (Noisy 또는 일반 레이어)"""
        if self.use_noisy:
            return nn.Sequential(
                NoisyLinear(input_size, 256),
                nn.ReLU(inplace=True),
                NoisyLinear(256, output_size)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, output_size)
            )

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, state: dict, return_features: bool = False) -> torch.Tensor:
        """순전파"""
        image = state['image']
        position = state['position']

        # CNN으로 이미지 처리
        cnn_features = self.cnn(image)

        # 위치 정보 인코딩
        pos_features = self.position_encoder(position)

        # 특징 융합
        combined = torch.cat([cnn_features, pos_features], dim=1)
        features = self.feature_fusion(combined)

        # Attention 적용
        attended_features = self.attention(features)

        if return_features:
            return attended_features

        # Categorical DQN vs Standard DQN
        if self.use_categorical:
            return self._categorical_forward(attended_features)
        else:
            return self._standard_forward(attended_features)

    def _standard_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Standard DQN forward pass"""
        if self.config.USE_DUELING_DQN:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(features)
        return q_values

    def _categorical_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Categorical DQN forward pass"""
        batch_size = features.size(0)

        if self.config.USE_DUELING_DQN:
            # Value stream
            value_logits = self.value_head(features)  # [batch, num_atoms]
            value_dist = F.softmax(value_logits, dim=1)

            # Advantage stream
            advantage_logits = self.advantage_head(features)  # [batch, action_size * num_atoms]
            advantage_logits = advantage_logits.view(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            advantage_dist = F.softmax(advantage_logits, dim=2)

            # Dueling aggregation for distributions
            value_dist = value_dist.unsqueeze(1).expand(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
            q_dist = value_dist + advantage_dist - advantage_mean

            # Convert distribution to Q-values (expectation)
            q_values = torch.sum(q_dist * self.support.view(1, 1, -1), dim=2)
        else:
            # Standard categorical DQN
            q_logits = self.q_head(features)  # [batch, action_size * num_atoms]
            q_logits = q_logits.view(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            q_dist = F.softmax(q_logits, dim=2)

            # Convert distribution to Q-values (expectation)
            q_values = torch.sum(q_dist * self.support.view(1, 1, -1), dim=2)

        return q_values

    def get_categorical_distribution(self, state: dict) -> torch.Tensor:
        """Get probability distributions for Categorical DQN"""
        if not self.use_categorical:
            raise ValueError("Network is not configured for categorical DQN")

        image = state['image']
        position = state['position']

        # Feature extraction
        cnn_features = self.cnn(image)
        pos_features = self.position_encoder(position)
        combined = torch.cat([cnn_features, pos_features], dim=1)
        features = self.feature_fusion(combined)
        attended_features = self.attention(features)

        batch_size = attended_features.size(0)

        if self.config.USE_DUELING_DQN:
            value_logits = self.value_head(attended_features)
            value_dist = F.softmax(value_logits, dim=1)

            advantage_logits = self.advantage_head(attended_features)
            advantage_logits = advantage_logits.view(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            advantage_dist = F.softmax(advantage_logits, dim=2)

            value_dist = value_dist.unsqueeze(1).expand(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
            q_dist = value_dist + advantage_dist - advantage_mean
        else:
            q_logits = self.q_head(attended_features)
            q_logits = q_logits.view(batch_size, self.config.ACTION_SIZE, self.num_atoms)
            q_dist = F.softmax(q_logits, dim=2)

        return q_dist

    def compute_curiosity_loss(self, current_state: dict, next_state: dict, actions: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic curiosity loss"""
        if not hasattr(self, 'curiosity'):
            return torch.tensor(0.0, device=actions.device)

        # Extract features
        current_features = self.forward(current_state, return_features=True)
        next_features = self.forward(next_state, return_features=True)

        # Curiosity forward pass
        pred_actions, pred_next_features = self.curiosity(current_features, next_features, actions)

        # Inverse model loss (action prediction)
        inverse_loss = F.cross_entropy(pred_actions, actions)

        # Forward model loss (next state prediction)
        forward_loss = F.mse_loss(pred_next_features, next_features.detach())

        # Total curiosity loss
        curiosity_loss = inverse_loss + forward_loss

        return curiosity_loss

    def update_dropout_rate(self, progress: float):
        """Update dropout rate based on training progress"""
        if self.config.USE_DROPOUT_SCHEDULING:
            # Linear interpolation between initial and final dropout rates
            current_rate = self.config.INITIAL_DROPOUT + progress * (self.config.FINAL_DROPOUT - self.config.INITIAL_DROPOUT)
            self.dropout.p = current_rate

    def reset_noise(self):
        """Noisy 레이어의 노이즈 리셋"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class MultiStepDQN(nn.Module):
    """Multi-step DQN 래퍼"""

    def __init__(self, network: DQNNetwork, n_steps: int = 3):
        super(MultiStepDQN, self).__init__()
        self.network = network
        self.n_steps = n_steps

    def forward(self, state: dict) -> torch.Tensor:
        return self.network(state)

    def get_multi_step_target(self, rewards: torch.Tensor, next_q_values: torch.Tensor,
                             dones: torch.Tensor, gamma: float) -> torch.Tensor:
        """Multi-step 타겟 계산"""
        multi_step_return = torch.zeros_like(rewards[:, 0])

        for i in range(self.n_steps):
            if i < rewards.shape[1]:
                multi_step_return += (gamma ** i) * rewards[:, i] * (~dones[:, i])

        # 마지막 스텝의 Q 값 추가
        multi_step_return += (gamma ** self.n_steps) * next_q_values * (~dones[:, -1])

        return multi_step_return

class EnsembleDQN(nn.Module):
    """앙상블 DQN (불확실성 추정)"""

    def __init__(self, config: Config, num_networks: int = 3):
        super(EnsembleDQN, self).__init__()
        self.networks = nn.ModuleList([
            DQNNetwork(config) for _ in range(num_networks)
        ])
        self.num_networks = num_networks

    def forward(self, state: dict, return_uncertainty: bool = False):
        q_values_list = []

        for network in self.networks:
            q_values = network(state)
            q_values_list.append(q_values)

        q_values_tensor = torch.stack(q_values_list)  # (num_networks, batch, actions)

        if return_uncertainty:
            mean_q = q_values_tensor.mean(dim=0)
            uncertainty = q_values_tensor.std(dim=0)
            return mean_q, uncertainty
        else:
            return q_values_tensor.mean(dim=0)

    def reset_noise(self):
        """모든 네트워크의 노이즈 리셋"""
        for network in self.networks:
            network.reset_noise()

def create_dqn_network(config: Config, network_type: str = "standard") -> nn.Module:
    """DQN 네트워크 팩토리 함수"""
    if network_type == "standard":
        return DQNNetwork(config, use_noisy=config.USE_NOISY_NETS)
    elif network_type == "multistep":
        base_network = DQNNetwork(config, use_noisy=config.USE_NOISY_NETS)
        return MultiStepDQN(base_network, config.MULTI_STEP_N)
    elif network_type == "ensemble":
        return EnsembleDQN(config)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

if __name__ == "__main__":
    # 네트워크 테스트
    config = Config()
    config.validate_config()

    print("DQN 네트워크 테스트 시작...")

    # 표준 DQN 테스트
    network = create_dqn_network(config, "standard")
    print(f"네트워크 파라미터 수: {sum(p.numel() for p in network.parameters()):,}")

    # 더미 입력 생성
    batch_size = config.BATCH_SIZE
    dummy_state = {
        'image': torch.randn(batch_size, config.IMAGE_CHANNELS,
                           config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        'position': torch.randn(batch_size, config.POSITION_DIM)
    }

    # 순전파 테스트
    with torch.no_grad():
        q_values = network(dummy_state)
        print(f"Q 값 출력 형태: {q_values.shape}")
        print(f"Q 값 범위: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")

    # GPU 메모리 사용량 확인 (CUDA 사용 시)
    if torch.cuda.is_available():
        network = network.cuda()
        dummy_state = {k: v.cuda() for k, v in dummy_state.items()}

        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        q_values = network(dummy_state)

        mem_after = torch.cuda.memory_allocated()
        print(f"GPU 메모리 사용량: {(mem_after - mem_before) / 1024**2:.2f} MB")

    print("네트워크 테스트 완료!")