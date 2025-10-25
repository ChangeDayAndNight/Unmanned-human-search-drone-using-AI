"""
DJI TELLO EDU 드론을 사용한 사람 탐지 시스템
YOLOv5n 모델을 이용하여 실시간으로 사람을 탐지합니다.
"""

import cv2
import numpy as np
import torch
from djitellopy import Tello
import time
import os
import warnings
import sys

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# YOLOv5 직접 사용을 위한 torch.hub 설정
torch.hub.set_dir('./models')  # 모델 저장 디렉토리 설정


class TelloHumanDetector:
    def __init__(self, model_path="best.pt"):
        """
        드론 및 YOLO 모델 초기화

        Args:
            model_path (str): YOLOv5 모델 가중치 파일 경로
        """
        # YOLO 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # YOLOv5 커스텀 모델 로드
        try:
            print(f"로컬 YOLOv5 모델 파일 로드 시도: {model_path}")

            # PyTorch의 weights_only 문제 해결을 위한 설정
            import functools
            original_load = torch.load
            torch.load = functools.partial(original_load, weights_only=False)

            try:
                # YOLOv5 GitHub에서 다운로드하려고 시도
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True, _verbose=False)
                print(f"YOLOv5 커스텀 모델 로드 성공: {model_path}")

            except Exception as hub_error:
                print(f"네트워크 오류로 인한 로드 실패: {hub_error}")
                raise Exception("네트워크 연결이 필요합니다. YOLOv5 repo를 로컬에 배치하거나 인터넷 연결을 확인해주세요.")

            # torch.load 원래대로 복구
            torch.load = original_load

        except Exception as e:
            print(f"로컬 모델 로드 실패: {e}")
            raise FileNotFoundError(f"모델 파일을 로드할 수 없습니다: {model_path}")
        
        print(f"YOLOv5 모델 로드 완료: {model_path}")

        # TELLO 드론 초기화
        self.tello = Tello()
        self.tello.connect()
        print(f"드론 배터리: {self.tello.get_battery()}%")

        # Keep-alive 기능 활성화 (자동 착륙 방지)
        print("자동 착륙 방지를 위한 Keep-alive 기능 활성화")

        # 드론 스트림 활성화
        self.tello.streamon()
        print("드론 스트림 시작")

        # 설정 변수
        self.TARGET_HEIGHT = 500  # 목표 고도 (cm) - takeoff 포함 총 고도
        self.DETECTION_CONFIDENCE = 0.5  # 탐지 신뢰도 임계값
        self.MIN_BATTERY = 20  # 최소 배터리 수준 (%)
        self.HOVER_DURATION = 120  # 감시 모드 시작 후 호버링 시간 (초)
        self.KEEP_ALIVE_INTERVAL = 2.0  # Keep-alive 명령 전송 간격 (초)
        self.frame_count = 0  # 프레임 카운터
        self.info_update_interval = 30  # 드론 정보 업데이트 간격 (프레임)
        self.last_keep_alive = time.time()  # 마지막 keep-alive 전송 시간

    def takeoff_and_hover(self):
        """
        드론을 이륙시키고 목표 고도까지 상승
        """
        # 배터리 확인
        battery = self.tello.get_battery()
        if battery < self.MIN_BATTERY:
            raise Exception(f"배터리 부족: {battery}% (최소 {self.MIN_BATTERY}% 필요)")

        print("드론 이륙 시작...")
        self.tello.takeoff()
        time.sleep(1)

        # 현재 고도 확인
        current_height = self.tello.get_height()
        print(f"이륙 후 현재 고도: {current_height}cm")

        # 목표 고도까지 상승
        if current_height < self.TARGET_HEIGHT:
            rise_distance = self.TARGET_HEIGHT - current_height
            print(f"{rise_distance}cm 더 상승하여 총 {self.TARGET_HEIGHT}cm 고도로 이동 중...")
            self.tello.move_up(rise_distance)
            time.sleep(1)

        final_height = self.tello.get_height()
        print(f"목표 고도에 도달: {final_height}cm, 호버링 중...")

        # 초기 Keep-alive 명령 전송
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            print("Keep-alive 명령 초기화 완료")
        except Exception as e:
            print(f"Keep-alive 초기화 실패: {e}")

    def detect_humans(self, frame):
        """
        프레임에서 사람을 탐지하고 바운딩 박스를 그립니다.

        Args:
            frame: OpenCV 이미지 프레임

        Returns:
            tuple: (처리된 프레임, 탐지된 사람 수)
        """
        # YOLOv5 모델로 객체 탐지 수행 (RGB 프레임 사용)
        results = self.model(frame, size=640)

        human_count = 0
        processed_frame = frame.copy()

        # YOLOv5 결과 처리 (pandas DataFrame 형식)
        try:
            detections = results.pandas().xyxy[0]  # pandas DataFrame

            for _, detection in detections.iterrows():
                confidence = detection['confidence']
                class_name = detection['name']

                # 신뢰도가 임계값보다 높고 사람인 경우만 처리
                if confidence >= self.DETECTION_CONFIDENCE and class_name == 'person':
                    human_count += 1

                    # 바운딩 박스 좌표 추출
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

                    # 바운딩 박스 그리기
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 신뢰도 라벨 추가
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"탐지 처리 오류: {e}")

        # 탐지된 사람 수 표시
        info_text = f"Detected Humans: {human_count}"
        cv2.putText(processed_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 드론 정보 표시 (일정 간격으로만 업데이트)
        if not hasattr(self, 'cached_battery') or self.frame_count % self.info_update_interval == 0:
            try:
                self.cached_battery = self.tello.get_battery()
                self.cached_height = self.tello.get_height()

            except Exception as e:
                print(f"드론 정보 업데이트 실패: {e}")

                if not hasattr(self, 'cached_battery'):
                    self.cached_battery = 0
                    self.cached_height = 0

        return processed_frame, human_count

    def start_detection_stream(self):
        """
        실시간 스트림에서 사람 탐지 시작 (1분간 호버링 포함)
        """
        print("실시간 사람 탐지 시작...")
        print("1분간 호버링 후 자유 비행 모드로 전환")
        print("종료하려면 'q' 키를 누르세요.")

        # 호버링 시작 시간 기록
        hover_start_time = time.time()
        hover_mode = True
        self.last_keep_alive = time.time()  # Keep-alive 시간 초기화

        try:
            while True:
                # 현재 시간 확인
                current_time = time.time()
                elapsed_time = current_time - hover_start_time

                # Keep-alive 명령 전송 (자동 착륙 방지)
                if current_time - self.last_keep_alive >= self.KEEP_ALIVE_INTERVAL:
                    try:
                        # 현재 상태를 유지하기 위한 명령 전송
                        if hover_mode:
                            # 호버링 모드: 정지 명령
                            self.tello.send_rc_control(0, 0, 0, 0)
                        else:
                            # 자유 비행 모드: 여전히 정지 상태 유지
                            self.tello.send_rc_control(0, 0, 0, 0)

                        self.last_keep_alive = current_time

                        # 디버깅: keep-alive 로그 (너무 빈번하지 않게 10초마다)
                        if int(current_time) % 10 == 0:
                            mode_status = "[호버링]" if hover_mode else "[자유비행]"
                            print(f"Keep-alive 명령 전송 {mode_status}")

                    except Exception as keep_alive_error:
                        print(f"Keep-alive 명령 전송 실패: {keep_alive_error}")
                        # Keep-alive 실패시에도 시간 업데이트
                        self.last_keep_alive = current_time

                # 호버링 모드 체크 (1분 = 60초)
                if hover_mode and elapsed_time >= self.HOVER_DURATION:
                    hover_mode = False
                    print(f"\n1분 호버링 완료! 이제 자유 비행 모드로 전환합니다.")

                # 드론 카메라에서 프레임 가져오기
                try:
                    frame = self.tello.get_frame_read().frame
                except Exception as frame_error:
                    print(f"프레임 가져오기 오류: {frame_error}")
                    continue

                if frame is None:
                    print("프레임이 None입니다. 드론 연결을 확인하세요.")
                    print("드론 연결 끄어짐으로 인한 종료...")
                    break

                # BGR에서 RGB로 변환 (YOLO 모델은 RGB를 기대함)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 프레임 카운터 증가
                self.frame_count += 1

                # 사람 탐지 수행 (RGB 프레임 사용)
                try:
                    processed_frame, human_count = self.detect_humans(frame_rgb)
                except Exception as detection_error:
                    print(f"탐지 처리 오류: {detection_error}")
                    processed_frame = frame_rgb
                    human_count = 0

                # 프레임 출력
                try:
                    cv2.imshow('TELLO Human Detection', processed_frame)
                except Exception as display_error:
                    print(f"화면 표시 오류: {display_error}")
                    print("화면 표시를 위해 OpenCV 창을 다시 열어보세요.")

                # 탐지 결과 로그 (30프레임마다 또는 사람 탐지 시)
                if human_count > 0 or self.frame_count % 30 == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    if human_count > 0:
                        mode_status = "[호버링]" if hover_mode else "[자유비행]"

                    else:
                        if hover_mode:
                            remaining_hover = max(0, self.HOVER_DURATION - elapsed_time)
                            print(f"[{timestamp}] [호버링] 감시 중... (남은 시간: {int(remaining_hover)}초, 배터리: {getattr(self, 'cached_battery', 'N/A')}%)")
                        else:
                            print(f"[{timestamp}] [자유비행] 감시 중... (배터리: {getattr(self, 'cached_battery', 'N/A')}%)")

                # 배터리 부족 시 자동 착륙
                if hasattr(self, 'cached_battery') and self.cached_battery < self.MIN_BATTERY:
                    print(f"배터리 부족 ({self.cached_battery}%)으로 인한 자동 착륙을 시작합니다.")
                    break

                # 'q' 키로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("사용자가 'q' 키를 눌러 종료합니다.")
                    break
                elif key != 255:  # 다른 키가 눌렸을 때
                    print(f"예상치 못한 키 입력: {key} (종료하려면 'q' 키를 누르세요)")

        except KeyboardInterrupt:
            print("\n사용자에 의한 중단 (Ctrl+C)...")

        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("\n=== 감시 루프 종료 ===\n")
            self.cleanup()

    def cleanup(self):
        """
        리소스 정리 및 드론 착륙
        """
        print("시스템 종료 중...")

        # OpenCV 창 닫기
        cv2.destroyAllWindows()

        # 드론 착륙
        try:
            print("드론 착륙 중...")
            self.tello.land()
            time.sleep(3)
        except Exception as e:
            print(f"착륙 중 오류: {e}")

        # 드론 스트림 종료
        try:
            self.tello.streamoff()
            print("드론 스트림 종료")
        except Exception as e:
            print(f"스트림 종료 중 오류: {e}")

        print("시스템 종료 완료")


def main():
    """
    메인 실행 함수
    """
    detector = None

    try:
        print("=== DJI TELLO EDU 사람 탐지 시스템 ===")
        print("시스템 초기화 중...")

        # 모델 경로 설정 (현재 디렉토리의 best.pt)
        model_path = "best.pt"

        # 드론 탐지 시스템 초기화
        print("1. YOLO 모델 및 드론 초기화...")
        detector = TelloHumanDetector(model_path)

        # 드론 이륙 및 목표 고도 상승
        print("2. 드론 이륙 및 고도 조정...")
        detector.takeoff_and_hover()

        # 실시간 탐지 시작
        print("3. 실시간 사람 탐지 시작...")
        detector.start_detection_stream()

    except FileNotFoundError as e:
        print(f"모델 파일 오류: {e}")
        print("best.pt 파일이 현재 디렉토리에 있는지 확인해주세요.")

    except Exception as e:
        print(f"시스템 오류: {e}")
        print("드론 연결을 확인하고 다시 시도해주세요.")

    finally:
        # 정리 작업
        if detector is not None:
            try:
                detector.cleanup()

            except Exception as e:
                print(f"정리 작업 중 오류: {e}")

        print("프로그램 종료")

if __name__ == "__main__":
    main()