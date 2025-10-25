import airsim
import cv2
import numpy as np
import os
import time
import math
from datetime import datetime

class DroneCamera:
    def __init__(self, save_path="C:\\Users\\cktmd\\Desktop\\Claude\\IMG_Capture", 
                 yolo_size=640, target_altitude=0, target_x=0, target_y=0,
                 auto_save_interval=1.0):
        """
        AirSim 드론 카메라 초기화
        
        Args:
            save_path: 이미지 저장 경로
            yolo_size: YOLOv5 학습용 이미지 크기 (기본 640x640)
            target_altitude: 목표 고도 (Z축, 기본 -50)
            target_x: 목표 X 좌표 (이동하면서 촬영)
            target_y: 목표 Y 좌표 (이동하면서 촬영)
            auto_save_interval: 자동 저장 간격 (초 단위, 기본 1.0초)
        """
        # 저장 경로 설정 및 생성
        self.save_path = save_path
        self.yolo_size = yolo_size
        self.auto_save_interval = auto_save_interval
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"[INFO] 저장 경로 생성: {self.save_path}")
        
        # AirSim 클라이언트 연결
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("[INFO] AirSim 연결 성공!")
        
        # 드론 활성화 및 Arm
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[INFO] 드론 API 제어 활성화 및 Arm 완료")
        
        # 이미지 카운터 및 자동 저장 타이머
        self.image_count = 0
        self.last_save_time = 0  # 마지막 저장 시간
        
        # 출발 위치 저장
        self.start_position = None
        
        # 드론 이동 시퀀스 실행s
        self.initialize_drone_position(target_altitude, target_x, target_y)
        
    def initialize_drone_position(self, altitude, x, y):
        """
        드론 이륙 및 목표 고도로 이동 후 즉시 x,y 이동 시작
        
        Args:
            altitude: 목표 고도 (Z축, 음수 값)
            x: 목표 X 좌표
            y: 목표 Y 좌표
        """
        print("\n" + "="*60)
        print("[STEP 1] 드론 이륙 시작")
        
        # 1단계: 이륙
        self.client.takeoffAsync().join()
        print("[SUCCESS] 이륙 완료")
        time.sleep(1)  # 안정화 대기
        
        # 출발 위치 저장
        state = self.client.getMultirotorState()
        self.start_position = state.kinematics_estimated.position
        
        # 2단계: 고도 이동 (Z축)
        print(f"\n[STEP 2] 고도 {altitude}m로 이동 중...")
        self.client.moveToZAsync(altitude, velocity=10).join()
        print(f"[SUCCESS] 고도 이동 완료 (Z: {altitude}m)")
        time.sleep(1)  # 짧은 안정화
        
        # 현재 위치 확인
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        # 3단계: 즉시 x, y 좌표로 이동 시작 (비동기 - join 없음)
        print(f"\n[STEP 3] 목표 좌표로 이동 시작: X = {x}, Y = {y}")
        self.client.moveToPositionAsync(x, y, altitude, velocity=3)
        # join() 없음 → 즉시 다음 단계로 (촬영 시작)
        
        # 이동 정보 저장
        self.target_x = x
        self.target_y = y
        self.target_altitude = altitude
        self.reached_target = False
        self.returning_home = False
        
    def get_camera_image(self):
        """
        드론 카메라에서 이미지 가져오기 (비압축 방식)
        AirSim 1.8.1은 실제로 BGR 형식으로 반환함

        Returns:
            tuple: (BGR 형식의 이미지, 원본 해상도), 실패 시 (None, None)
        """
        try:
            # 비압축 배열로 이미지 요청
            # False, False = pixels_as_float=False, compress=False
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            # 응답 유효성 검증
            if not responses or len(responses) == 0:
                return None, None

            response = responses[0]

            # 이미지 데이터 존재 확인
            if response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
                return None, None

            # 너비와 높이 확인
            if response.width == 0 or response.height == 0:
                return None, None

            # 원본 해상도 정보
            original_resolution = (response.width, response.height)

            # 1D 배열을 numpy array로 변환
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            # AirSim은 문서와 달리 실제로 BGR (3채널) 형식으로 반환
            # reshape: (height, width, 3) - BGR 순서
            img_bgr = img1d.reshape(response.height, response.width, 3)

            # BGR 그대로 반환 (변환 불필요)
            return img_bgr, original_resolution

        except:
            return None, None
    
    def save_image(self, image, original_resolution=None):
        """
        이미지를 YOLOv5 학습용 크기로 리사이즈하여 저장

        Args:
            image: 저장할 이미지 (numpy array, BGR 형식)
            original_resolution: 원본 해상도 (width, height)
        """
        
        # YOLOv5 학습용 크기로 리사이즈 (정사각형)
        # INTER_CUBIC 사용으로 더 나은 품질
        resized_image = cv2.resize(image, (self.yolo_size, self.yolo_size),
                                   interpolation=cv2.INTER_CUBIC)

        # 타임스탬프로 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"drone_img_{timestamp}_{self.yolo_size}x{self.yolo_size}.png"
        filepath = os.path.join(self.save_path, filename)

        # BGR 이미지를 그대로 저장 (cv2.imwrite는 BGR을 기대)
        cv2.imwrite(filepath, resized_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # 최고 품질
        self.image_count += 1
    
    def check_reached_target(self, current_pos):
        """
        목표 지점 도달 여부 확인
        
        Args:
            current_pos: 현재 위치
            
        Returns:
            bool: 도달 여부
        """
        distance = math.sqrt(
            (current_pos.x_val - self.target_x) ** 2 + 
            (current_pos.y_val - self.target_y) ** 2
        )
        return distance < 3.0  # 3m 이내면 도달로 판단
        
    def run_stream(self):
        """
        실시간 카메라 스트림 실행 및 제어 (1초 간격 자동 저장)
        드론은 이미 이동 중인 상태에서 촬영 시작
        """
        
        cv2.namedWindow("AirSim Drone Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AirSim Drone Camera", 1920, 1080)
        
        # 초기 연결 대기 (빠른 체크)
        retry_count = 0
        max_retries = 5  # 빠르게 체크

        while retry_count < max_retries:
            test_image, _ = self.get_camera_image()
            if test_image is not None:
                break
            retry_count += 1
            time.sleep(0.3)  # 짧은 대기
        
        if retry_count >= max_retries:
            print("[ERROR] 카메라 초기화 실패. Unreal Engine 상태를 확인하세요.")
            return
        
        # 드론 이동 상태 확인
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"\n[INFO] 목표 위치: X = {self.target_x}, Y = {self.target_y}")
        
        # 자동 저장 타이머 초기화
        self.last_save_time = time.time()
        
        try:
            frame_count = 0
            while True:
                # 카메라 이미지 가져오기 (BGR 형식)
                image_bgr, original_resolution = self.get_camera_image()

                if image_bgr is None:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # 현재 드론 위치 확인
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position
                
                # 실시간 위치 정보 업데이트 (이스케이프 문자 사용)
                print()
                print(f"\033[2A\033[K[드론 위치] X: {pos.x_val:7.2f} | Y: {pos.y_val:7.2f} | Z: {pos.z_val:7.2f} | 저장: {self.image_count}장")
                
                # 자동 저장 (1초 간격)
                elapsed_time = current_time - self.last_save_time
                if elapsed_time >= self.auto_save_interval:
                    self.save_image(image_bgr, original_resolution)
                    self.last_save_time = current_time
                
                # 목표 도달 확인 및 복귀
                if not self.reached_target and not self.returning_home:
                    if self.check_reached_target(pos):
                        self.reached_target = True
                        print("\n[INFO] 목표 지점 도달!")
                        
                        # 현재 yaw 값 가져오기
                        current_orientation = state.kinematics_estimated.orientation
                        current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                        current_yaw_deg = math.degrees(current_yaw)
                        
                        # 180도 회전
                        target_yaw = current_yaw_deg + 180
                        # -180 ~ 180 범위로 정규화
                        if target_yaw > 180:
                            target_yaw -= 360
                        elif target_yaw < -180:
                            target_yaw += 360
                        
                        print(f"[INFO] 180도 회전 (현재: {current_yaw_deg:.1f}° → 목표: {target_yaw:.1f}°)")
                        self.client.rotateToYawAsync(target_yaw, timeout_sec=5, margin=5).join()

                        time.sleep(3)

                        print("[INFO] 회전 완료! 출발 지점으로 복귀")
                        
                        # 출발 지점으로 복귀
                        self.client.moveToPositionAsync(
                            self.start_position.x_val,
                            self.start_position.y_val,
                            self.target_altitude,
                            velocity=3
                        )
                        self.returning_home = True
                
                # 출발 지점 복귀 완료 확인
                if self.returning_home:
                    distance_home = math.sqrt(
                        (pos.x_val - self.start_position.x_val) ** 2 + 
                        (pos.y_val - self.start_position.y_val) ** 2
                    )
                    if distance_home < 2.5:
                        print("\n[INFO] 출발 지점 복귀 완료!")
                        print("[INFO] 'Q' 키를 눌러 종료하세요.\n")
                        self.returning_home = False  # 반복 방지
                
                # 화면에 표시 (BGR 그대로, 변환 불필요)
                cv2.imshow("AirSim Drone Camera", image_bgr)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s') or key == ord('S'):
                    # 수동 저장 (즉시)
                    self.save_image(image_bgr, original_resolution)
                    
                elif key == ord('q') or key == ord('Q'):
                    # 종료
                    print("\n[INFO] 프로그램 종료 중...")
                    break
                    
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C 감지 - 프로그램 종료")
            
        finally:
            # 리소스 정리
            cv2.destroyAllWindows()
            self.cleanup()
            self.client.reset()
            
    def cleanup(self):
        """
        드론 제어 해제 및 연결 종료
        """
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("\n[INFO] 드론 제어 해제 완료")
        print(f"[완료] 총 {self.image_count}장의 이미지가 저장되었습니다.")


def main():
    """
    메인 실행 함수
    """
    try:
        # 설정 값
        SAVE_PATH = "C:\\Users\\cktmd\\Desktop\\Claude\\IMG_Capture\\IMG"
        YOLO_SIZE = 1280  # YOLOv5 학습용 이미지 크기 (640, 1280, 1920 가능)
        TARGET_ALTITUDE = -25 # 목표 고도 (Z축)
        TARGET_X = 25  # 목표 X 좌표 
        TARGET_Y = -300  # 목표 Y 좌표
        AUTO_SAVE_INTERVAL = 2.0  # 자동 저장 간격 (초)
        
        print("="*60)
        print("             AirSim 드론 카메라 캡처 시스템")
        print("="*60 + "\n")
        
        # 드론 카메라 객체 생성 및 자동 이동
        drone_cam = DroneCamera(
            save_path=SAVE_PATH,
            yolo_size=YOLO_SIZE,
            target_altitude=TARGET_ALTITUDE,
            target_x=TARGET_X,
            target_y=TARGET_Y,
            auto_save_interval=AUTO_SAVE_INTERVAL
        )
        
        # 스트림 실행
        drone_cam.run_stream()
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {str(e)}")
        print("[TIP] Unreal Engine에서 AirSim이 실행 중인지 확인해주세요.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()