#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
거리 기반 Out of Bounds 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
import math

def test_distance_calculation():
    """거리 기반 경계 이탈 로직 테스트"""
    print("=== 거리 기반 Out of Bounds 테스트 ===")

    # 테스트 설정
    config = Config()
    target_pos = config.TARGET_POSITIONS[0]  # [20, 20, -10]
    print(f"목표 위치: {target_pos}")

    # 시뮬레이션된 드론 시작 위치
    start_pos = [0, 0, -10]
    print(f"시작 위치: {start_pos}")

    # 초기 거리 계산
    initial_distance = math.sqrt(
        (start_pos[0] - target_pos[0]) ** 2 +
        (start_pos[1] - target_pos[1]) ** 2 +
        (start_pos[2] - target_pos[2]) ** 2
    )
    print(f"초기 목표 거리: {initial_distance:.2f}m")

    # 거리 임계값 (3배)
    distance_threshold = initial_distance * 3.0
    print(f"경계 이탈 임계값: {distance_threshold:.2f}m")

    # 테스트 케이스들
    test_cases = [
        ([5, 5, -10], "목표에 가까운 위치"),
        ([40, 40, -10], "목표에서 멀지만 임계값 내"),
        ([80, 80, -10], "경계 이탈 예상 위치"),
        ([-50, -50, -10], "반대 방향 경계 이탈"),
        ([100, 0, -10], "한 축만 멀리 이동")
    ]

    print("\n=== 테스트 케이스 실행 ===")
    for pos, description in test_cases:
        current_distance = math.sqrt(
            (pos[0] - target_pos[0]) ** 2 +
            (pos[1] - target_pos[1]) ** 2 +
            (pos[2] - target_pos[2]) ** 2
        )

        is_out_of_bounds = current_distance > distance_threshold
        status = "경계 이탈" if is_out_of_bounds else "정상"

        print(f"{description}: {pos}")
        print(f"  현재 거리: {current_distance:.2f}m")
        print(f"  상태: {status}")
        print()

    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    test_distance_calculation()