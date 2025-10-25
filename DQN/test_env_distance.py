#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경에서 거리 기반 경계 이탈 기능 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
import math

def simulate_distance_bounds_check():
    """거리 기반 경계 이탈 시뮬레이션"""
    print("=== 환경 거리 기반 경계 이탈 시뮬레이션 ===")

    config = Config()

    # 시뮬레이션된 환경 변수들
    target_positions = config.TARGET_POSITIONS
    current_target_idx = 0
    target_pos = target_positions[current_target_idx]

    # 시뮬레이션된 시작 위치와 초기 거리
    start_position = [0, 0, -10]
    initial_distance = math.sqrt(
        (start_position[0] - target_pos[0]) ** 2 +
        (start_position[1] - target_pos[1]) ** 2 +
        (start_position[2] - target_pos[2]) ** 2
    )

    print(f"목표 위치: {target_pos}")
    print(f"시작 위치: {start_position}")
    print(f"초기 거리: {initial_distance:.2f}m")
    print(f"경계 이탈 임계값: {initial_distance * 3.0:.2f}m")

    def is_out_of_bounds_simulation(position):
        """시뮬레이션된 _is_out_of_bounds 함수"""
        if initial_distance is None:
            return False

        current_distance_to_target = math.sqrt(
            (position[0] - target_pos[0]) ** 2 +
            (position[1] - target_pos[1]) ** 2 +
            (position[2] - target_pos[2]) ** 2
        )

        distance_threshold = initial_distance * 3.0
        is_too_far = current_distance_to_target > distance_threshold

        if is_too_far:
            print(f"거리 기반 경계 이탈 - 현재: {current_distance_to_target:.2f}m, 임계값: {distance_threshold:.2f}m")

        return is_too_far

    # 테스트 시나리오
    test_positions = [
        [10, 10, -10],   # 목표에 가까움
        [50, 50, -10],   # 적당히 멀지만 안전
        [70, 70, -10],   # 경계선 근처
        [85, 85, -10],   # 경계 이탈 (거의 3배)
        [-60, -60, -10], # 반대 방향 경계 이탈
    ]

    print("\n=== 테스트 시나리오 ===")
    for i, pos in enumerate(test_positions, 1):
        print(f"\n시나리오 {i}: {pos}")
        is_out = is_out_of_bounds_simulation(pos)
        print(f"결과: {'경계 이탈' if is_out else '정상'}")

    print("\n=== 기존 고정 경계와 비교 ===")
    map_bounds = config.MAP_BOUNDS
    print(f"기존 고정 경계: X({map_bounds['x_min']}~{map_bounds['x_max']}), Y({map_bounds['y_min']}~{map_bounds['y_max']})")
    print(f"새로운 거리 기반: 목표 중심 반경 {initial_distance * 3.0:.2f}m")

    print("\n=== 장점 ===")
    print("1. 목표 위치에 따라 동적으로 경계가 조정됨")
    print("2. 드론이 목표에서 너무 멀어지는 것을 방지")
    print("3. 학습 효율성 향상 (목표 중심 탐색)")

if __name__ == "__main__":
    simulate_distance_bounds_check()