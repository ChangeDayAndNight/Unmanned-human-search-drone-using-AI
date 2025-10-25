import airsim
import cv2
import numpy as np
import os
import time
import math
import torch
import warnings
from datetime import datetime

# PyTorch FutureWarning 및 기타 경고 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*untrusted repository.*")

# 환경 변수 설정으로 추가 경고 억제
os.environ['PYTHONWARNINGS'] = 'ignore'

class YOLODroneController:
    def __init__(self, model_path="Unreal_350.pt",
                 target_x=0, target_y=-200, target_z=-25):
        """
        AirSim 드론 컨트롤러 with YOLOv5 객체 탐지

        Args:
            model_path: YOLOv5 모델 가중치 경로
            target_x: 목표 X 좌표
            target_y: 목표 Y 좌표
            target_z: 목표 Z 좌표 (고도)
        """
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z

        # YOLOv5 모델 로드
        print("[INFO] YOLOv5 모델 로딩 중...")
        try:
            # 추가 경고 억제
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                          path=model_path, force_reload=True)
            self.model.conf = 0.5  # 신뢰도 임계값
            self.model.iou = 0.45   # NMS IoU 임계값
            print("[SUCCESS] YOLOv5 모델 로드 완료")
        except Exception as e:
            print(f"[ERROR] YOLOv5 모델 로드 실패: {e}")
            print("[INFO] 모델 없이 실행됩니다.")
            self.model = None

        # AirSim 클라이언트 연결
        print("[INFO] AirSim 연결 중...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("[SUCCESS] AirSim 연결 완료")

        # 드론 활성화
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[INFO] 드론 제어 활성화 완료")

        # 통계
        self.frame_count = 0
        self.detection_count = 0

    def move_to_target_async(self):
        """드론을 목표 좌표로 비동기 이동 시작"""
        print(f"\n[STEP 1] 드론 이륙...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-25, 7.5).join()
        print("[SUCCESS] 이륙 완료")

        print(f"[STEP 2] 목표 좌표로 이동 시작: X={self.target_x}, Y={self.target_y}, Z={self.target_z}")
        # 비동기 이동 시작 (join() 없음)
        self.move_task = self.client.moveToPositionAsync(
            self.target_x, self.target_y, self.target_z, velocity=3
        )
        print("[INFO] 이동 중... 실시간 영상 시작")

        # 이동 상태 플래그
        self.is_moving = True
        self.target_reached = False

    def get_camera_image(self):
        """드론 카메라에서 이미지 획득"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if not responses or len(responses) == 0:
                return None

            response = responses[0]
            if response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
                return None

            # 1D 배열을 BGR 이미지로 변환
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgr = img1d.reshape(response.height, response.width, 3)

            return img_bgr

        except Exception as e:
            print(f"[ERROR] 이미지 획득 실패: {e}")
            return None

    def detect_objects(self, image):
        """YOLOv5로 객체 탐지"""
        if self.model is None:
            return image, []

        try:
            # YOLOv5 추론 (경고 억제)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.model(image)

            # 탐지 결과 파싱
            detections = results.pandas().xyxy[0].to_dict(orient="records")

            # 결과 이미지에 바운딩 박스 그리기
            annotated_image = image.copy()

            for detection in detections:
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                                 int(detection['xmax']), int(detection['ymax'])
                confidence = detection['confidence']
                class_name = detection['name']

                # 바운딩 박스 그리기
                color = (0, 255, 0)  # 초록색
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 라벨 텍스트
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # 라벨 배경
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)

                # 라벨 텍스트
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            return annotated_image, detections

        except Exception as e:
            print(f"[ERROR] 객체 탐지 실패: {e}")
            return image, []

    def check_movement_status(self):
        """이동 상태 확인 및 업데이트"""
        if hasattr(self, 'is_moving') and self.is_moving:
            # 현재 위치 확인
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position

            # 목표 지점과의 거리 계산
            distance = math.sqrt(
                (pos.x_val - self.target_x) ** 2 +
                (pos.y_val - self.target_y) ** 2 +
                (pos.z_val - self.target_z) ** 2
            )

            # 목표 지점 도달 확인 (3m 이내)
            if distance < 3.0:
                self.is_moving = False
                self.target_reached = True
                print(f"\n[SUCCESS] 목표 좌표 도달! (거리: {distance:.1f}m)")
                return True

        return False

    def run_real_time_detection(self):
        """실시간 영상 스트림 및 객체 탐지"""
        print("\n[INFO] 실시간 객체 탐지 시작")
        print("[CONTROL] 'Q' 키로 종료")

        # OpenCV 창 설정
        cv2.namedWindow("YOLOv5 Real-time Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv5 Real-time Detection", 1080, 720)

        # 초기 카메라 연결 확인
        retry_count = 0
        while retry_count < 10:
            test_image = self.get_camera_image()
            if test_image is not None:
                print("[SUCCESS] 카메라 연결 확인")
                # 초기 이미지 표시
                cv2.imshow("YOLOv5 Real-time Detection", test_image)
                cv2.waitKey(1)
                break
            retry_count += 1
            print(f"[INFO] 카메라 연결 시도 중... ({retry_count}/10)")
            time.sleep(0.5)

        if retry_count >= 10:
            print("[ERROR] 카메라 연결 실패")
            return

        # 메인 루프
        try:
            start_time = time.time()

            while True:
                # 카메라 이미지 획득
                image = self.get_camera_image()
                if image is None:
                    time.sleep(0.1)
                    continue

                self.frame_count += 1

                # 드론 위치 정보
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position

                # 이동 상태 확인
                movement_completed = self.check_movement_status()

                # YOLOv5 객체 탐지
                annotated_image, detections = self.detect_objects(image)

                if detections:
                    self.detection_count += len(detections)

                # 목표 지점과의 거리 계산
                if hasattr(self, 'target_x'):
                    distance_to_target = math.sqrt(
                        (pos.x_val - self.target_x) ** 2 +
                        (pos.y_val - self.target_y) ** 2 +
                        (pos.z_val - self.target_z) ** 2
                    )
                else:
                    distance_to_target = 0

                # 이동 상태 표시
                if hasattr(self, 'is_moving') and self.is_moving:
                    movement_status = f"Moving to target... ({distance_to_target:.1f}m)"
                elif hasattr(self, 'target_reached') and self.target_reached:
                    movement_status = "Target reached!"
                else:
                    movement_status = "Ready"

                # 영상 출력
                cv2.imshow("YOLOv5 Real-time Detection", annotated_image)

                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

                # FPS 제한 (optional)
                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C로 종료")

        finally:
            # 정리
            cv2.destroyAllWindows()
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n[INFO] 시스템 정리 중...")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()

        # 통계 출력
        print(f"[STATS] 총 프레임: {self.frame_count}")
        print(f"[STATS] 총 탐지 객체: {self.detection_count}")
        print("[INFO] 정리 완료")


def main():
    """메인 실행 함수"""
    try:
        print("="*60)
        print("        AirSim YOLOv5 실시간 객체 탐지 시스템")
        print("="*60)

        # YOLOv5 드론 컨트롤러 생성
        controller = YOLODroneController(
            model_path="best.pt",
            target_x=0,    # 요청된 좌표
            target_y=-200,  # 요청된 좌표
            target_z=-25   # 요청된 좌표
        )

        # 드론 이동 시작 (비동기)
        controller.move_to_target_async()

        # 실시간 탐지 시작 (이동과 동시에)
        controller.run_real_time_detection()

    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        print("[TIP] Unreal Engine에서 AirSim이 실행 중인지 확인하세요.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()