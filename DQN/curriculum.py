"""
최적화 커리큘럼 학습 설정
확장된 학습 시간을 활용한 심화 학습
"""

from config import Config

class CurriculumConfig(Config):
    """최적화 커리큘럼 학습용 설정"""

    def __init__(self):
        super().__init__()

        # 5단계 확장 커리큘럼
        self.NUM_EPISODES = 1000  # 에피소드 수

        self.CURRICULUM_STAGES = {
            # Stage 1: 기본 비행 안정화 (0-200)
            'basic_stabilization': {
                'episodes': (0, 200),
                'focus': 'flight_stability_and_collision_avoidance',
                'max_episode_steps': 150,  # 기본 학습 시간
                'target_radius': 12.0,  # 넓은 반경으로 쉬운 시작
                'rewards': {
                    'target_reached': 100.0,     # 높은 초기 보상
                    'collision': -150.0,         # 강한 충돌 페널티
                    'getting_closer': 2.5,       # 높은 접근 보상
                    'exploration_bonus': 1.2,    # 높은 탐색 보상
                    'time_penalty': -0.05,       # 낮은 시간 페널티
                    'stability_bonus': 8.0,      # 안정성 보너스
                }
            },

            # Stage 2: 기본 목표 탐지 (200-400)
            'basic_target_detection': {
                'episodes': (200, 400),
                'focus': 'single_target_reaching_mastery',
                'max_episode_steps': 250,
                'target_radius': 8.0,  # 중간 반경
                'rewards': {
                    'target_reached': 120.0,     # 목표 도달 보상
                    'collision': -120.0,         # 중간 충돌 페널티
                    'getting_closer': 2.0,       # 중간 접근 보상
                    'exploration_bonus': 0.8,    # 중간 탐색 보상
                    'time_penalty': -0.08,       # 중간 시간 페널티
                    'precision_bonus': 12.0,     # 정밀도 보너스
                }
            },

            # Stage 3: 정밀 내비게이션 (400-600)
            'precision_navigation': {
                'episodes': (400, 600),
                'focus': 'accurate_path_planning',
                'max_episode_steps': 350,
                'target_radius': 6.0,  # 표준 반경
                'rewards': {
                    'target_reached': 150.0,     # 높은 목표 보상
                    'collision': -100.0,         # 중간 충돌 페널티
                    'getting_closer': 1.8,       # 중간 접근 보상
                    'exploration_bonus': 0.5,    # 낮은 탐색 보상
                    'time_penalty': -0.12,       # 중간 시간 페널티
                    'precision_bonus': 18.0,     # 높은 정밀도 보너스
                    'efficiency_bonus': 15.0,    # 효율성 보너스
                }
            },

            # Stage 4: 다중 목표 처리 (600-850)
            'multi_target_handling': {
                'episodes': (600, 850),
                'focus': 'efficient_multi_target_search',
                'max_episode_steps': 500,
                'target_radius': 4.0,  # 작은 반경
                'rewards': {
                    'target_reached': 180.0,     # 매우 높은 목표 보상
                    'collision': -80.0,          # 낮은 충돌 페널티 (학습된 회피)
                    'getting_closer': 1.5,       # 낮은 접근 보상
                    'exploration_bonus': 0.3,    # 매우 낮은 탐색 보상
                    'time_penalty': -0.15,       # 높은 시간 페널티
                    'efficiency_bonus': 25.0,    # 높은 효율성 보너스
                    'multi_target_bonus': 40.0,  # 다중 목표 보너스
                    'sequence_bonus': 20.0,      # 순차 달성 보너스
                }
            },

            # Stage 5: 마스터 레벨 최적화 (850-1000)
            'master_optimization': {
                'episodes': (850, 1000),
                'focus': 'optimal_strategy_refinement',
                'max_episode_steps': 750,  # 최대 길이
                'target_radius': 3.0,  # 매우 작은 반경
                'rewards': {
                    'target_reached': 220.0,     # 최고 목표 보상
                    'collision': -60.0,          # 매우 낮은 충돌 페널티
                    'getting_closer': 1.2,       # 최소 접근 보상
                    'exploration_bonus': 0.1,    # 최소 탐색 보상
                    'time_penalty': -0.2,        # 최고 시간 페널티
                    'efficiency_bonus': 35.0,    # 최고 효율성 보너스
                    'multi_target_bonus': 60.0,  # 최고 다중 목표 보너스
                    'sequence_bonus': 30.0,      # 최고 순차 달성 보너스
                    'mastery_bonus': 50.0,       # 마스터 보너스
                }
            }
        }

        # 현재 스테이지 추적
        self.current_stage = 'basic_stabilization'

    def get_stage_config(self, episode: int):
        """현재 에피소드에 맞는 스테이지 설정 반환"""
        for stage_name, stage_config in self.CURRICULUM_STAGES.items():
            start_ep, end_ep = stage_config['episodes']
            if start_ep <= episode < end_ep:
                self.current_stage = stage_name
                return stage_name, stage_config

        # 마지막 스테이지 반환
        return 'master_optimization', self.CURRICULUM_STAGES['master_optimization']

    def update_config_for_episode(self, episode: int):
        """에피소드에 따라 설정 동적 조정"""
        stage_name, stage_config = self.get_stage_config(episode)

        # 동적 설정 업데이트
        self.MAX_EPISODE_STEPS = stage_config['max_episode_steps']
        self.TARGET_RADIUS = stage_config['target_radius']
        self.REWARDS.update(stage_config['rewards'])

        return stage_name, stage_config

    def get_learning_rate_schedule(self, episode: int):
        """최신 기법 최적화된 학습률 스케줄 (Categorical DQN + Curiosity)"""
        if episode < 100:
            return 0.0003   # Categorical DQN에 맞게 조정
        elif episode < 250:
            return 0.0002   # 안정적인 분포 학습
        elif episode < 500:
            return 0.00015  # 중간 학습률
        elif episode < 750:
            return 0.0001   # 미세 조정
        else:
            return 0.00008  # 최종 안정화

    def get_epsilon_schedule(self, episode: int):
        """점진적 epsilon 스케줄"""
        if episode < 150:
            # 초기 탐험 단계
            return max(0.5, 1.0 - episode * 0.0033)
        elif episode < 400:
            # 중간 탐험 감소
            return max(0.25, 0.5 - (episode - 150) * 0.001)
        elif episode < 700:
            # 점진적 감소
            return max(0.1, 0.25 - (episode - 400) * 0.0005)
        else:
            # 최종 미세 조정
            return max(0.05, 0.1 - (episode - 700) * 0.000167)

    def get_target_update_frequency(self, episode: int):
        """동적 타겟 업데이트 주기"""
        if episode < 200:
            return 150  # 빠른 초기 업데이트
        elif episode < 500:
            return 200  # 중간 업데이트
        elif episode < 800:
            return 250  # 안정적 업데이트
        else:
            return 300  # 최종 안정화

    def get_curriculum_info(self, episode: int):
        """현재 커리큘럼 정보 반환"""
        stage_name, stage_config = self.get_stage_config(episode)

        progress_in_stage = episode - stage_config['episodes'][0]
        total_stage_episodes = stage_config['episodes'][1] - stage_config['episodes'][0]
        stage_progress = progress_in_stage / total_stage_episodes

        overall_progress = episode / self.NUM_EPISODES

        return {
            'stage': stage_name,
            'focus': stage_config['focus'],
            'stage_progress': stage_progress,
            'overall_progress': overall_progress,
            'episodes_in_stage': f"{progress_in_stage}/{total_stage_episodes}",
            'learning_rate': self.get_learning_rate_schedule(episode),
            'epsilon': self.get_epsilon_schedule(episode),
            'target_update_freq': self.get_target_update_frequency(episode)
        }

# 확장 마일스톤
CURRICULUM_MILESTONES = {
    100: {
        'target': '기본 비행 안정성 확립',
        'metrics': ['충돌률 < 30%', '평균 비행시간 > 50 스텝'],
        'expected_reward': 20
    },
    200: {
        'target': '안정적 비행 완성',
        'metrics': ['충돌률 < 20%', '탐색 커버리지 > 30%'],
        'expected_reward': 60
    },
    350: {
        'target': '단일 목표 도달 숙련',
        'metrics': ['목표 도달률 > 50%', '평균 거리 감소 > 20%'],
        'expected_reward': 100
    },
    500: {
        'target': '정밀 내비게이션 완성',
        'metrics': ['목표 도달률 > 70%', '탐색 효율성 > 60%'],
        'expected_reward': 140
    },
    650: {
        'target': '다중 목표 기본 능력',
        'metrics': ['목표 도달률 > 75%', '평균 목표 수 > 1.5'],
        'expected_reward': 170
    },
    800: {
        'target': '다중 목표 숙련',
        'metrics': ['목표 도달률 > 85%', '평균 목표 수 > 2.5'],
        'expected_reward': 200
    },
    950: {
        'target': '마스터 레벨 달성',
        'metrics': ['목표 도달률 > 90%', '평균 목표 수 > 3.0', '효율성 > 85%'],
        'expected_reward': 240
    },
    1000: {
        'target': '완전한 자율 비행 마스터',
        'metrics': ['목표 도달률 > 95%', '평균 목표 수 > 3.5', '안정성 > 95%'],
        'expected_reward': 280
    }
}

def print_curriculum_progress(episode: int, config: CurriculumConfig):
    """커리큘럼 진행상황 출력"""
    info = config.get_curriculum_info(episode)

    print(f"\n커리큘럼 진행 (Episode {episode})")
    print(f"현재 단계: {info['stage']}")
    print(f"포커스: {info['focus']}")
    print(f"단계 진행률: {info['stage_progress']:.1%} ({info['episodes_in_stage']})")
    print(f"전체 진행률: {info['overall_progress']:.1%}")
    print(f"학습률: {info['learning_rate']:.5f}")
    print(f"Epsilon: {info['epsilon']:.3f}")
    print(f"타겟 업데이트 주기: {info['target_update_freq']}")

    # 마일스톤 체크
    if episode in CURRICULUM_MILESTONES:
        milestone = CURRICULUM_MILESTONES[episode]
        print(f"\n마일스톤 {episode}: {milestone['target']}")
        print("목표 지표:")
        for metric in milestone['metrics']:
            print(f"  - {metric}")
        print(f"예상 보상: {milestone['expected_reward']}")

    # 진행도 바 표시
    progress_bar = "=" * int(info['overall_progress'] * 30)
    empty_bar = "-" * (30 - len(progress_bar))
    print(f"진행도: [{progress_bar}{empty_bar}] {info['overall_progress']:.1%}")

def get_stage_recommendations(episode: int, config: CurriculumConfig):
    """각 단계별 학습 권장사항"""
    stage_name, stage_config = config.get_stage_config(episode)

    recommendations = {
        'basic_stabilization': [
            "안정적인 비행 패턴 개발에 집중하세요",
            "충돌 회피 능력을 완전히 마스터하세요",
            "기본 제어 명령들의 효과를 학습하세요",
            "넓은 탐험을 통해 환경을 이해하세요"
        ],
        'basic_target_detection': [
            "단일 목표 도달 정확도를 향상시키세요",
            "효율적인 경로 계획을 개발하세요",
            "목표 인식 능력을 강화하세요",
            "시간 효율성을 고려하기 시작하세요"
        ],
        'precision_navigation': [
            "정밀한 목표 도달에 집중하세요",
            "복잡한 경로 계획을 학습하세요",
            "장애물 회피와 목표 도달의 균형을 맞추세요",
            "효율성과 정확성을 동시에 추구하세요"
        ],
        'multi_target_handling': [
            "다중 목표 처리 전략을 개발하세요",
            "목표 간 이동 최적화에 집중하세요",
            "순차적 목표 달성 능력을 향상시키세요",
            "복잡한 시나리오에 대응하세요"
        ],
        'master_optimization': [
            "완벽한 자율 비행 능력을 완성하세요",
            "모든 상황에서의 최적 전략을 개발하세요",
            "마스터 레벨의 효율성을 달성하세요",
            "실전 배포를 위한 최종 준비를 완료하세요"
        ]
    }

    return recommendations.get(stage_name, ["지속적인 개선을 위해 노력하세요"])

if __name__ == "__main__":
    # 커리큘럼 설정 테스트
    config = CurriculumConfig()

    test_episodes = [0, 150, 300, 500, 700, 900, 1000]

    print("최적화 커리큘럼 테스트")
    print("=" * 70)

    for ep in test_episodes:
        print_curriculum_progress(ep, config)

        # 권장사항 출력
        recommendations = get_stage_recommendations(ep, config)
        print("\n이 단계 권장사항:")
        for i, rec in enumerate(recommendations[:2], 1):  # 상위 2개만
            print(f"  {i}. {rec}")

        print("-" * 60)

    print(f"\n총 {len(CURRICULUM_MILESTONES)}개 마일스톤으로 완전한 학습!")
    print("마스터 레벨 자율 비행 능력 완성 예정!")