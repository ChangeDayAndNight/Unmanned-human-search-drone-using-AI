"""
AirSim Environment Wrapper for DQN Drone Navigation
무인 인명 탐색 드론을 위한 강화학습 환경
"""

import airsim
import numpy as np
import cv2
import time
import math
import random
from typing import Tuple, List, Dict, Optional
from config import Config

class AirSimDroneEnv:
    """AirSim 기반 드론 강화학습 환경"""

    def __init__(self):
        self.config = Config()
        self.client = None
        self.current_target_idx = 0
        self.visited_targets = set()
        self.episode_step = 0
        self.previous_distance = float('inf')
        self.start_position = [0, 0, -10]  # 시작 위치
        self.initial_distance_to_target = None  # 초기 목표 거리

        # 탐색 맵 최적화 (크기 축소)
        self.exploration_map = np.zeros((100, 100), dtype=np.uint8)  # 100x100 그리드, uint8로 메모리 절약
        self.exploration_resolution = 2.0  # 2미터 해상도로 확대

        # 충돌 검증 최적화
        self.collision_history = []
        self.last_valid_position = None
        self.last_collision_time = 0
        self.previous_velocity = [0.0, 0.0, 0.0]  # 속도 기반 충돌 감지용

        # 행동 다양성 추적
        self.action_history = []
        self.consecutive_stop_count = 0

        self._connect_to_airsim()

    def _connect_to_airsim(self):
        """AirSim 연결 전용 (이륙은 reset에서)"""
        try:
            self.client = airsim.MultirotorClient(
                ip=self.config.AIRSIM_HOST,
                port=self.config.AIRSIM_PORT
            )
            self.client.confirmConnection()
            print("[SUCCESS] AirSim 연결 성공!")

        except Exception as e:
            print(f"[ERROR] AirSim 연결 실패: {e}")
            raise

    def reset(self) -> np.ndarray:
        """환경 리셋 (타임아웃 및 robust 에러 핸들링)"""
        self.episode_step = 0
        self.visited_targets = set()
        self.exploration_map.fill(0)

        # 충돌 검증 변수들 리셋
        self.collision_history.clear()
        self.last_valid_position = None
        self.last_collision_time = 0
        self.previous_velocity = [0.0, 0.0, 0.0]  # 속도 기반 충돌 감지 리셋

        # 행동 추적 변수들 리셋
        self.action_history.clear()
        self.consecutive_stop_count = 0

        try:
            # 에피소드 리셋

            # 1. AirSim 환경 리셋 (타임아웃 적용)
            try:
                self.client.reset()
                # AirSim 환경 리셋 완료
            except Exception as e:
                # AirSim 리셋 실패 무시
                pass

            # 2. 드론 재초기화 및 이륙 (robust 처리)
            try:
                self.client.enableApiControl(True, self.config.VEHICLE_NAME)
                self.client.armDisarm(True, self.config.VEHICLE_NAME)

                # 드론 재이륙
                takeoff_result = self.client.takeoffAsync(vehicle_name=self.config.VEHICLE_NAME)

                # 타임아웃 적용 (5초)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(takeoff_result.join)
                    try:
                        future.result(timeout=5.0)  # 5초 타임아웃
                    except concurrent.futures.TimeoutError:
                        # 이륙 타임아웃
                        pass

                time.sleep(0.5)  # 안정화 시간 단축
            except Exception as e:
                # 드론 이륙 실패 무시
                pass

            # 3. 랜덤 시작 위치 설정
            self.start_position = [
                random.uniform(-15, 15),  # X 범위
                random.uniform(-15, 15),  # Y 범위
                random.uniform(-12, -6)   # 안전한 비행 고도 (6-12미터)
            ]

            # 4. 랜덤 목표 선택
            self.current_target_idx = random.randint(0, len(self.config.TARGET_POSITIONS) - 1)

            # 새로운 에피소드 시작

            # 5. 드론을 새로운 시작 위치로 이동 (타임아웃 적용)
            try:
                move_result = self.client.moveToPositionAsync(
                    self.start_position[0],
                    self.start_position[1],
                    self.start_position[2],
                    velocity=8,  # 더 빠른 이동
                    vehicle_name=self.config.VEHICLE_NAME
                )

                # 타임아웃 적용 (3초)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(move_result.join)
                    try:
                        future.result(timeout=3.0)  # 3초 타임아웃
                    except concurrent.futures.TimeoutError:
                        pass  # Timeout occurred, continue

            except Exception as e:
                print(f"⚠️ 위치 이동 실패, 기본 위치 사용: {e}")

            # 6. 호버링 상태로 안정화 (타임아웃 적용)
            try:
                hover_result = self.client.hoverAsync(vehicle_name=self.config.VEHICLE_NAME)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(hover_result.join)
                    try:
                        future.result(timeout=2.0)  # 2초 타임아웃
                    except concurrent.futures.TimeoutError:
                        print("⚠️ 호버링 명령 타임아웃, 계속 진행")

                time.sleep(0.2)  # 최소 안정화 시간
            except Exception as e:
                print(f"⚠️ 호버링 실패, 계속 진행: {e}")

            # 초기 거리 계산
            try:
                self.previous_distance = self._calculate_distance_to_target()
                self.initial_distance_to_target = self.previous_distance  # 초기 거리 저장
            except Exception as e:
                print(f"⚠️ 거리 계산 실패, 기본값 사용: {e}")
                self.previous_distance = 100.0  # 기본값
                self.initial_distance_to_target = 100.0  # 초기 거리 기본값

            # Episode reset completed silently

        except Exception as e:
            print(f"❌ 환경 리셋 중 심각한 오류: {e}")
            # 최소한의 안전 처리
            try:
                self.client.hoverAsync(vehicle_name=self.config.VEHICLE_NAME)
                time.sleep(0.5)
            except:
                pass

            # 기본값으로 설정
            self.start_position = [0, 0, -10]
            self.current_target_idx = 0
            self.previous_distance = 100.0
            self.initial_distance_to_target = 100.0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행"""
        self.episode_step += 1

        try:
            # 행동 추적 (행동 다양성 분석용)
            self.action_history.append(action)
            if len(self.action_history) > 10:  # 최근 10개 행동만 보존
                self.action_history.pop(0)

            # 연속 동일 행동 추적 (새로운 Action Space에는 정지가 없음)
            if len(self.action_history) >= 2 and action == self.action_history[-1] == self.action_history[-2]:
                self.consecutive_stop_count += 1
            else:
                self.consecutive_stop_count = 0

            # 첫 몇 스텝은 조기 종료 방지
            if self.episode_step <= 3:
                # 초기 스텝에서는 충돌 검사 완화
                pass

            # 액션 실행
            self._execute_action(action)

            # 상태 획득
            state = self._get_state()

            # 보상 계산
            reward, done, info = self._calculate_reward()

            # 첫 5스텝 내에서는 조기 종료 방지 (충돌 제외)
            if self.episode_step <= 5 and info.get('reason') not in ['collision']:
                done = False
                if info.get('reason') == 'out_of_bounds':
                    # 경계 이탈 시 드론을 안전한 공중 위치로 이동
                    safe_pos = [0, 0, -10]  # 중앙 상공 10미터
                    move_result = self.client.moveToPositionAsync(
                        safe_pos[0], safe_pos[1], safe_pos[2],
                        velocity=10,  # 빠르게 이동
                        vehicle_name=self.config.VEHICLE_NAME
                    )
                    move_result.join()
                    self.client.hoverAsync(vehicle_name=self.config.VEHICLE_NAME).join()
                    reward = -10  # 가벼운 페널티
                    info['reason'] = 'repositioned'
                    print("⚠️ 경계 이탈 - 안전 위치로 복귀")

            return state, reward, done, info

        except Exception as e:
            print(f"⚠️ 스텝 실행 중 오류: {e}")
            # 안전한 기본값 반환
            state = self._get_state()
            return state, -1.0, False, {'reason': 'error', 'error': str(e)}

    def _execute_action(self, action: int):
        """액션을 드론 제어 명령으로 변환하여 실행"""
        if action >= len(self.config.ACTIONS):
            action = 4  # 잘못된 액션은 전진으로 처리 (가장 안전한 기본 행동)

        velocity = self.config.ACTIONS[action]

        # 속도 명령 실행
        self.client.moveByVelocityAsync(
            velocity[0], velocity[1], velocity[2],
            self.config.MOVEMENT_DURATION,
            vehicle_name=self.config.VEHICLE_NAME
        )

        # 명령 완료 대기 (안정적인 행동 실행을 위한 대기)
        action_wait_time = max(0.2, self.config.MOVEMENT_DURATION * 0.8)  # 최소 200ms 보장
        time.sleep(action_wait_time)

    def _get_state(self) -> np.ndarray:
        """현재 상태 반환 (이미지 + 위치 정보) - 최적화됨"""
        # RGB 이미지 획득
        image = self._get_camera_image()

        # 드론 상태 정보 획득 (한 번만 호출)
        drone_state = self.client.getMultirotorState(self.config.VEHICLE_NAME)
        position = drone_state.kinematics_estimated.position

        # 위치 정보 최적화 (오일러 각도 변환 생략, 쿼터니언 직접 사용)
        orientation = drone_state.kinematics_estimated.orientation

        # 위치 정보 정규화 (벡터화된 연산)
        pos_data = np.array([position.x_val, position.y_val, position.z_val], dtype=np.float32)
        pos_normalized = np.clip(pos_data / [100, 100, 50], -1, 1)

        # 간단한 방향 정보 (쿼터니언의 z, w 성분만 사용)
        orientation_simplified = np.array([orientation.z_val, orientation.w_val], dtype=np.float32)
        orientation_normalized = np.clip(orientation_simplified, -1, 1)

        # 목표까지의 상대 위치 (벡터화)
        target_pos = np.array(self.config.TARGET_POSITIONS[self.current_target_idx], dtype=np.float32)
        relative_pos = (target_pos - pos_data) / [100, 100, 50]
        relative_pos_normalized = np.clip(relative_pos, -1, 1)

        # 위치 벡터 구성 (8차원으로 축소) - x,y,z, qz,qw, rel_x,rel_y,rel_z
        position_vector = np.concatenate([
            pos_normalized,           # 3차원
            orientation_normalized,   # 2차원
            relative_pos_normalized   # 3차원
        ], dtype=np.float32)

        # 이미지 전처리
        image = self._preprocess_image(image)

        return {
            'image': image,
            'position': position_vector
        }

    def _get_camera_image(self) -> np.ndarray:
        """드론 카메라에서 RGB 이미지 획득"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], self.config.VEHICLE_NAME)

            if responses:
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                return img_rgb
            else:
                # 기본 이미지 반환
                return np.zeros((self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3), dtype=np.uint8)

        except Exception as e:
            print(f"이미지 획득 오류: {e}")
            return np.zeros((self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3), dtype=np.uint8)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (크기 조정, 정규화)"""
        # 크기 조정
        resized = cv2.resize(image, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))

        # 정규화 (0-1 범위)
        normalized = resized.astype(np.float32) / 255.0

        # 채널 순서 변경 (H, W, C) -> (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))

        return transposed

    def _validate_collision(self, collision_info, current_pos, velocity) -> bool:
        """강화된 충돌 검증: AirSim의 충돌 정보를 적극적으로 활용"""
        if not collision_info.has_collided:
            return False

        current_time = time.time()

        # 1. 충돌 위치 검증 (AirSim에서 제공하는 충돌 지점)
        collision_pos = collision_info.position
        impact_point = [collision_pos.x_val, collision_pos.y_val, collision_pos.z_val]

        # 충돌 지점이 현재 위치와 합리적인 거리 내에 있는지 확인
        collision_distance = math.sqrt(
            (current_pos[0] - impact_point[0])**2 +
            (current_pos[1] - impact_point[1])**2 +
            (current_pos[2] - impact_point[2])**2
        )

        # 충돌 지점이 너무 멀면 false positive
        if collision_distance > 5.0:
            return False

        # 2. 충돌 법선벡터 강화 검증
        normal = collision_info.normal
        normal_magnitude = math.sqrt(normal.x_val**2 + normal.y_val**2 + normal.z_val**2)

        # 법선벡터 유의미성 검사
        if normal_magnitude < 0.05:
            return False

        # 3. 충돌 관입 깊이 검증 (AirSim에서 제공)
        penetration_depth = collision_info.penetration_depth
        if penetration_depth < 0.01:  # 1cm 미만 관입 무시
            return False

        # 4. 속도 기반 충돌 강도 검증
        velocity_magnitude = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)

        # 충돌 임팩트 계산 (속도 * 관입깊이)
        impact_force = velocity_magnitude * penetration_depth
        if impact_force < 0.05:  # 약한 충돌 무시
            return False

        # 5. 충돌 쿨다운 체크 (개선됨)
        if current_time - self.last_collision_time < 1.0:  # 1초 쿨다운
            return False

        # 6. 연속 충돌 방지 (드론이 물체에 끼였을 때)
        self.collision_history.append({
            'time': current_time,
            'position': impact_point.copy(),
            'penetration': penetration_depth,
            'velocity': velocity_magnitude
        })

        # 최근 3초간의 충돌 기록만 유지
        self.collision_history = [c for c in self.collision_history if current_time - c['time'] < 3.0]

        # 같은 위치에서 반복적인 충돌이 발생하면 무시 (끼임 상태)
        if len(self.collision_history) >= 3:
            recent_positions = [c['position'] for c in self.collision_history[-3:]]
            position_variance = np.var([
                math.sqrt(sum((p[i] - impact_point[i])**2 for i in range(3)))
                for p in recent_positions
            ])

            if position_variance < 1.0:  # 끼임 상태 감지
                return False

        # 7. 지면 충돌 특별 처리
        ground_threshold = -0.5  # 지면 기준 고도
        if current_pos[2] > ground_threshold:
            # 지면 근처에서는 수직 법선벡터만 실제 지면 충돌로 인정
            normalized_normal = [normal.x_val/normal_magnitude,
                               normal.y_val/normal_magnitude,
                               normal.z_val/normal_magnitude]

            # Z축 법선 0.9 이상이면 지면 충돌
            if abs(normalized_normal[2]) > 0.9:
                self.last_collision_time = current_time
                return True

        # 8. 실제 충돌 확정

        self.last_collision_time = current_time
        return True

    def _log_collision_debug_info(self, collision_info, current_pos, velocity):
        """충돌 디버깅 정보 출력"""
        try:
            velocity_magnitude = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            collision_pos = collision_info.position
            normal = collision_info.normal
            penetration = collision_info.penetration_depth

            print(f"[COLLISION DEBUG] 충돌 신호 감지:")
            print(f"   드론위치: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
            print(f"   충돌위치: ({collision_pos.x_val:.2f}, {collision_pos.y_val:.2f}, {collision_pos.z_val:.2f})")
            print(f"   속도: {velocity_magnitude:.3f} m/s")
            print(f"   관입깊이: {penetration:.6f} m")
            print(f"   법선벡터: ({normal.x_val:.3f}, {normal.y_val:.3f}, {normal.z_val:.3f})")
            print(f"   시간: {time.time():.3f}")

        except Exception as e:
            print(f"⚠️ 충돌 정보 출력 중 오류: {e}")
            pass

    def _detect_collision_with_sensors(self) -> bool:
        """센서 기반 충돌 감지 (거리 센서 활용)"""
        try:
            # 거리 센서 데이터 획득 (전방, 하방)
            distance_sensor_data = self.client.getDistanceSensorData("Distance", self.config.VEHICLE_NAME)

            # 센서 데이터가 유효한 경우
            if distance_sensor_data.distance > 0:
                # 너무 가까운 거리면 충돌 위험
                if distance_sensor_data.distance < 0.5:  # 50cm 이하
                    print(f"[SENSOR] 거리 센서 충돌 위험 감지: {distance_sensor_data.distance:.3f}m")
                    return True

            # 라이다 센서가 있다면 활용
            try:
                lidar_data = self.client.getLidarData("Lidar", self.config.VEHICLE_NAME)
                if lidar_data.point_cloud_np.size > 0:
                    # 포인트 클라우드에서 가까운 점들 확인
                    points = lidar_data.point_cloud_np.reshape(-1, 3)
                    min_distance = np.min(np.linalg.norm(points, axis=1))

                    if min_distance < 1.0:  # 1m 이하의 장애물
                        print(f"[SENSOR] 라이다 장애물 감지: {min_distance:.3f}m")
                        return True

            except (AttributeError, RuntimeError) as e:
                # 라이다 센서가 설정되지 않은 경우 무시
                pass

        except (ConnectionError, AttributeError, RuntimeError) as e:
            # 센서가 설정되지 않거나 연결 문제가 있는 경우 무시
            pass

        return False

    def _check_ground_collision(self, current_pos) -> bool:
        """지면 충돌 체크 (고도 기반)"""
        # AirSim의 지면 고도 정보 획득
        try:
            ground_height = self.client.simGetGroundTruthKinematics(self.config.VEHICLE_NAME).position.z_val

            # 현재 고도와 지면 거리 계산
            height_above_ground = abs(current_pos[2] - ground_height)

            # 지면에 너무 가까우면 충돌로 간주
            if height_above_ground < 0.3:  # 30cm 이하
                print(f"[GROUND] 지면 근접 충돌 감지 - 지면 거리: {height_above_ground:.3f}m")
                return True

        except Exception as e:
            # 기본적인 고도 체크
            if current_pos[2] > -0.5:  # 지면 기준 50cm 이하
                print(f"[GROUND] 기본 지면 충돌 감지 - 고도: {current_pos[2]:.3f}m")
                return True

        return False

    def _check_ground_collision_strict(self, current_pos) -> bool:
        """엄격한 지면 충돌 체크 (더 보수적)"""
        try:
            # AirSim의 지면 고도 정보 획득
            ground_height = self.client.simGetGroundTruthKinematics(self.config.VEHICLE_NAME).position.z_val

            # 현재 고도와 지면 거리 계산
            height_above_ground = abs(current_pos[2] - ground_height)

            # 지면에 매우 가까울 때만 충돌로 간주 (10cm 이하)
            if height_above_ground < 0.1:
                print(f"[GROUND] 엄격한 지면 충돌 감지 - 지면 거리: {height_above_ground:.3f}m")
                return True

        except Exception as e:
            # 기본적인 고도 체크 (더 엄격한 기준)
            if current_pos[2] > -0.2:  # 지면 기준 20cm 이하일 때만
                print(f"[GROUND] 엄격한 기본 지면 충돌 감지 - 고도: {current_pos[2]:.3f}m")
                return True

        return False

    def _calculate_reward(self) -> Tuple[float, bool, Dict]:
        """보상 함수 계산"""
        reward = 0.0
        done = False
        info = {'reason': ''}

        # 현재 위치 및 상태 획득
        drone_state = self.client.getMultirotorState(self.config.VEHICLE_NAME)
        position = drone_state.kinematics_estimated.position
        velocity = drone_state.kinematics_estimated.linear_velocity
        current_pos = [position.x_val, position.y_val, position.z_val]

        # 1. 매우 보수적인 충돌 검사 (False Positive 완전 제거)
        collision_info = self.client.simGetCollisionInfo(self.config.VEHICLE_NAME)

        # 기본적으로 충돌 없음으로 설정
        is_real_collision = False
        collision_reason = ""

        # AirSim collision 신호는 노이즈가 많으므로 매우 제한적으로만 사용
        if collision_info.has_collided:
            # 디버깅 정보 출력
            # 디버그 정보는 로깅 레벨에 따라 출력
            import logging
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                print(f"[COLLISION DEBUG] AirSim 충돌 신호 감지 - Step: {self.episode_step}")
                print(f"  관입깊이: {collision_info.penetration_depth:.6f}m")
                print(f"  충돌위치: ({collision_info.position.x_val:.2f}, {collision_info.position.y_val:.2f}, {collision_info.position.z_val:.2f})")
                print(f"  드론위치: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")

            # 극도로 엄격한 충돌 판정 - 관입깊이가 50cm 이상인 경우에만
            if collision_info.penetration_depth > 0.5:  # 50cm 이상 관입
                velocity_magnitude = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
                print(f"  속도: {velocity_magnitude:.3f}m/s")

                # 속도도 매우 빠른 경우에만 실제 충돌로 인정
                if velocity_magnitude > 3.0:  # 3m/s 이상의 고속 충돌
                    is_real_collision = True
                    collision_reason = "고속물리충돌"
                    print(f"[COLLISION] 극한 물리 충돌 감지됨!")
                else:
                    print(f"[COLLISION] 속도 부족으로 무시 ({velocity_magnitude:.3f}m/s < 3.0m/s)")
            else:
                print(f"[COLLISION] 관입깊이 부족으로 무시 ({collision_info.penetration_depth:.6f}m < 0.5m)")

        # 지면 충돌 검사 (매우 극단적인 경우에만)
        if not is_real_collision and current_pos[2] > 0:  # 지면 위에 있을 때만
            print(f"[COLLISION DEBUG] 지면 위 감지 - 고도: {current_pos[2]:.3f}m")
            is_real_collision = True
            collision_reason = "지면충돌"
            print(f"[COLLISION] 지면 충돌 감지됨!")

        # 속도 기반 충돌 감지 (급격한 속도 변화)
        if not is_real_collision and hasattr(self, 'previous_velocity'):
            velocity_magnitude = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            prev_vel_magnitude = math.sqrt(self.previous_velocity[0]**2 + self.previous_velocity[1]**2 + self.previous_velocity[2]**2)

            velocity_change = abs(velocity_magnitude - prev_vel_magnitude)

            # 급격한 속도 감소 (10m/s 이상 차이)
            if velocity_change > 10.0 and velocity_magnitude < 1.0:
                print(f"[COLLISION DEBUG] 급격한 속도 변화 - 이전: {prev_vel_magnitude:.3f}, 현재: {velocity_magnitude:.3f}")
                is_real_collision = True
                collision_reason = "급속도변화"
                print(f"[COLLISION] 급속도 변화로 충돌 감지됨!")

        # 이전 속도 저장
        self.previous_velocity = [velocity.x_val, velocity.y_val, velocity.z_val]

        if is_real_collision:
            reward += self.config.REWARDS['collision']
            done = True
            info['reason'] = 'collision'
            velocity_magnitude = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)

            print(f"[COLLISION CONFIRMED] 유형: {collision_reason} | 스텝: {self.episode_step} | 위치: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) | 속도: {velocity_magnitude:.2f}m/s")
            return reward, done, info

        # AirSim 충돌 신호가 있었지만 무시된 경우에는 별도 출력 없음 (이미 위에서 디버그 출력됨)

        # 유효한 위치 업데이트
        self.last_valid_position = current_pos.copy()

        # 2. 경계 이탈 검사
        if self._is_out_of_bounds(current_pos):
            reward += self.config.REWARDS['out_of_bounds']
            done = True
            info['reason'] = 'out_of_bounds'
            return reward, done, info

        # 3. 목표 도달 검사
        current_distance = self._calculate_distance_to_target()
        if current_distance < self.config.TARGET_RADIUS:
            reward += self.config.REWARDS['target_reached']
            self.visited_targets.add(self.current_target_idx)

            # 새로운 목표 설정 또는 에피소드 종료
            if len(self.visited_targets) >= len(self.config.TARGET_POSITIONS):
                done = True
                info['reason'] = 'all_targets_found'
            else:
                # 새로운 목표 선택
                remaining_targets = [i for i in range(len(self.config.TARGET_POSITIONS))
                                   if i not in self.visited_targets]
                self.current_target_idx = random.choice(remaining_targets)
                self.previous_distance = self._calculate_distance_to_target()
                self.initial_distance_to_target = self.previous_distance  # 새 목표에 대한 초기 거리 설정
                info['reason'] = 'target_reached'

        # 4. 재설계된 거리 기반 보상 시스템
        distance_change = self.previous_distance - current_distance
        if distance_change > 0:  # 목표에 가까워짐 (거리 감소)
            reward += self.config.REWARDS['getting_closer'] * distance_change
        elif distance_change < 0:  # 목표에서 멀어짐 (거리 증가) - 강력한 패널티
            reward += self.config.REWARDS['getting_farther'] * abs(distance_change)

        # 5. 기본 생존 보상 (매 스텝마다)
        reward += self.config.REWARDS['step_reward']

        # 6. 개선된 고도 기반 보상 시스템
        target_altitude = self.config.TARGET_POSITIONS[self.current_target_idx][2]
        current_altitude = current_pos[2]
        altitude_diff = abs(current_altitude - target_altitude)

        # 이전 고도 차이 추적 (고도 학습 개선)
        if not hasattr(self, 'previous_altitude_diff'):
            self.previous_altitude_diff = altitude_diff

        altitude_change = self.previous_altitude_diff - altitude_diff

        # 고도 차이 기반 지속적 보상/패널티
        if altitude_diff <= 1.5:  # 목표 고도 ±1.5m 이내 - 강한 보상
            reward += 2.0
        elif altitude_diff <= 3.0:  # 목표 고도 ±3m 이내 - 보상
            reward += 1.0
        elif altitude_diff <= 5.0:  # 목표 고도 ±5m 이내 - 약간 보상
            reward += 0.3
        else:  # 목표 고도에서 5m 이상 차이 - 거리 비례 패널티
            reward -= min(altitude_diff * 0.5, 5.0)  # 최대 -5.0 패널티

        # 고도 개선 보상 (목표 고도에 가까워지면 추가 보상)
        if altitude_change > 0.2:  # 고도가 목표에 가까워짐
            reward += min(altitude_change * 2.0, 3.0)  # 최대 +3.0 보상
        elif altitude_change < -0.2:  # 고도가 목표에서 멀어짐
            reward -= min(abs(altitude_change) * 1.5, 2.0)  # 최대 -2.0 패널티

        # 위험 고도 강한 패널티 (지면 근접 또는 과도한 고도)
        if current_pos[2] > -1.0:  # 지면 1m 이내
            reward -= 10.0
        elif current_pos[2] < -60.0:  # 60m 이상 고도
            reward -= 5.0

        # 이전 고도 차이 업데이트
        self.previous_altitude_diff = altitude_diff

        # 7. 행동 다양성 보상 시스템 (config.py와 일치)
        action = self.action_history[-1] if self.action_history else 0

        # 움직임 보상 (정지가 아닌 행동)
        if action != 0:
            reward += self.config.REWARDS.get('movement_reward', 0.1)

        # 연속 정지 패널티 (강화됨)
        if self.consecutive_stop_count >= 3:
            penalty_multiplier = min(self.consecutive_stop_count - 2, 5)
            reward += self.config.REWARDS.get('stop_penalty', -0.5) * penalty_multiplier

        # 8. 조건부 탐색 보상 (새로운 영역에서만)
        exploration_reward = self._calculate_exploration_reward(current_pos)
        if exploration_reward > 0:
            reward += self.config.REWARDS['exploration_bonus']

        # 9. 목표 진전 보상 (목표에 근접할 때)
        if current_distance < 10.0 and distance_change > 0:
            reward += self.config.REWARDS['progress_bonus']

        # 10. 효율성 보상 (빠른 목표 달성)
        max_steps = getattr(self.config, 'MAX_EPISODE_STEPS', 1000)
        if info.get('reason') == 'target_reached' and self.episode_step < max_steps * 0.5:
            reward += self.config.REWARDS['efficiency_bonus']

        # 11. 최대 스텝 검사
        if self.episode_step >= max_steps:
            done = True
            info['reason'] = 'max_steps'

        # Update previous distance
        self.previous_distance = current_distance

        # 보상 구성 요소 디버깅 정보 (매 50스텝마다)
        if self.episode_step % 50 == 0:
            reward_breakdown = {
                'distance_change': distance_change,
                'step_reward': self.config.REWARDS['step_reward'],
                'movement': self.config.REWARDS.get('movement_reward', 0.1) if action != 0 else 0,
                'exploration': self.config.REWARDS['exploration_bonus'] if exploration_reward > 0 else 0,
                'total': reward
            }

        # Additional info
        info.update({
            'distance_to_target': current_distance,
            'distance_change': distance_change,
            'targets_found': len(self.visited_targets),
            'current_target': self.current_target_idx,
            'episode_step': self.episode_step,
            'reward_total': reward
        })

        return reward, done, info

    def _calculate_distance_to_target(self) -> float:
        """현재 목표까지의 거리 계산"""
        drone_state = self.client.getMultirotorState(self.config.VEHICLE_NAME)
        position = drone_state.kinematics_estimated.position
        target_pos = self.config.TARGET_POSITIONS[self.current_target_idx]

        return math.sqrt(
            (position.x_val - target_pos[0]) ** 2 +
            (position.y_val - target_pos[1]) ** 2 +
            (position.z_val - target_pos[2]) ** 2
        )

    def _calculate_exploration_reward(self, position: List[float]) -> float:
        """탐색 보상 계산 (조건부 적용)"""
        # 그리드 좌표로 변환 (100x100 맵, 2m 해상도)
        grid_x = int((position[0] + 100) / self.exploration_resolution)
        grid_y = int((position[1] + 100) / self.exploration_resolution)

        # 경계 체크 및 새로운 영역 탐색시에만 보상 반환 (bool)
        if 0 <= grid_x < 100 and 0 <= grid_y < 100:
            if self.exploration_map[grid_x, grid_y] == 0:
                self.exploration_map[grid_x, grid_y] = 1
                return 1.0  # 새로운 영역 발견시만 1.0 반환

        return 0.0  # 이미 방문한 영역이거나 경계 밖

    def _is_out_of_bounds(self, position: List[float]) -> bool:
        """거리 기반 경계 이탈 검사: 초기 목표 거리의 3배 이상 멀어지면 종료"""
        if self.initial_distance_to_target is None:
            return False  # 초기 거리가 설정되지 않았으면 체크하지 않음

        # 현재 목표까지의 거리 계산
        target_pos = self.config.TARGET_POSITIONS[self.current_target_idx]
        current_distance_to_target = math.sqrt(
            (position[0] - target_pos[0]) ** 2 +
            (position[1] - target_pos[1]) ** 2 +
            (position[2] - target_pos[2]) ** 2
        )

        # 초기 거리의 3배를 초과하면 경계 이탈로 판정
        distance_threshold = self.initial_distance_to_target * 3.0
        is_too_far = current_distance_to_target > distance_threshold

        if is_too_far:
            print(f"🚫 거리 기반 경계 이탈 - 현재: {current_distance_to_target:.2f}m, 임계값: {distance_threshold:.2f}m")

        return is_too_far

    def get_current_position(self) -> List[float]:
        """현재 드론 위치 반환"""
        drone_state = self.client.getMultirotorState(self.config.VEHICLE_NAME)
        position = drone_state.kinematics_estimated.position
        return [position.x_val, position.y_val, position.z_val]

    def set_target_position(self, target_idx: int):
        """목표 위치 설정"""
        if 0 <= target_idx < len(self.config.TARGET_POSITIONS):
            self.current_target_idx = target_idx
            self.previous_distance = self._calculate_distance_to_target()

    def render(self) -> Dict:
        """환경 상태 시각화 정보 반환"""
        current_pos = self.get_current_position()
        target_pos = self.config.TARGET_POSITIONS[self.current_target_idx]

        return {
            'drone_position': current_pos,
            'target_position': target_pos,
            'distance_to_target': self._calculate_distance_to_target(),
            'visited_targets': list(self.visited_targets),
            'exploration_coverage': np.sum(self.exploration_map) / self.exploration_map.size
        }

    def close(self):
        """환경 정리"""
        if self.client:
            try:
                print("[GROUND] 드론 착륙 중...")

                # 드론 착륙
                land_result = self.client.landAsync(vehicle_name=self.config.VEHICLE_NAME)
                land_result.join()  # 착륙 완료까지 대기

                # 드론 비활성화
                self.client.armDisarm(False, self.config.VEHICLE_NAME)
                self.client.enableApiControl(False, self.config.VEHICLE_NAME)

                print("✅ 드론 착륙 완료 - AirSim 연결 해제")

            except Exception as e:
                print(f"⚠️ 드론 착륙 중 오류: {e}")
                # 강제로 비활성화
                try:
                    self.client.armDisarm(False, self.config.VEHICLE_NAME)
                    self.client.enableApiControl(False, self.config.VEHICLE_NAME)
                except (ConnectionError, RuntimeError, AttributeError) as e:
                    # AirSim 연결 해제 실패는 무시 (이미 연결이 끊어진 상태일 수 있음)
                    pass
                print("AirSim 연결 강제 해제")

            # 추가 리소스 정리
            self.client = None
            if hasattr(self, 'exploration_map'):
                self.exploration_map = None
            if hasattr(self, 'collision_history'):
                self.collision_history.clear()
            if hasattr(self, 'action_history'):
                self.action_history.clear()

    def __enter__(self):
        """Context manager 지원"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료 시 자동 정리"""
        self.close()

if __name__ == "__main__":
    # 환경 테스트
    print("[INFO] AirSim 드론 환경 테스트 시작...")

    env = AirSimDroneEnv()

    try:
        print("[INFO] 드론 이륙 및 환경 초기화 완료")
        state = env.reset()
        print(f"[INFO] 초기 상태 - 이미지: {state['image'].shape}, 위치: {state['position'].shape}")

        for i in range(10):
            action = random.randint(0, Config.ACTION_SIZE - 1)
            state, reward, done, info = env.step(action)

            # 현재 위치 출력
            current_pos = env.get_current_position()
            print(f"[STEP] {i+1}: 액션={action}, 보상={reward:.2f}, 위치=({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f}), 종료={done}")

            if done:
                print(f"[END] 에피소드 종료: {info['reason']}")
                break

        print("[INFO] 테스트 완료 - 드론 착륙 중...")

    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("[INFO] 환경 테스트 종료")