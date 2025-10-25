"""
Tello EDU 드론 360도 이미지 촬영 프로그램
5m 고도에서 30도씩 회전하며 각 각도마다 5개 이미지 촬영 및 저장
"""

from djitellopy import Tello
import cv2
import os
import time
from datetime import datetime
import threading
import glob
import re
from PIL import Image
import numpy as np

class TelloDroneCapture:
    def __init__(self):
        """드론 초기화 및 설정"""
        self.tello = Tello()
        self.save_path = r"C:\Users\cktmd\Desktop\TELLO_EDU\IMG_Caputre\captured_images"
        self.target_height = 500  # 5m = 500cm
        self.rotation_angle = 30  # 30도씩 회전
        self.wait_time = 30  # 각 위치에서 30초 대기
        self.frame = None
        self.streaming = False
        self.keep_alive_active = False

        # 저장 경로 생성
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"📁 저장 경로 생성: {self.save_path}")
    
    def connect_drone(self):
        """드론 연결 및 상태 확인"""
        try:
            self.tello.connect()
            print("✅ 드론 연결 성공")
            
            # 배터리 확인
            battery = self.tello.get_battery()
            print(f"🔋 배터리 잔량: {battery}%")
            
            if battery < 20:
                print("⚠️ 배터리 부족! 충전 필요")
                return False
                
            # 온도 확인
            temp = self.tello.get_temperature()
            print(f"🌡️ 드론 온도: {temp}°C")
            
            # keep-alive 스레드 시작
            self.start_keep_alive()

            return True

        except Exception as e:
            print(f"❌ 드론 연결 실패: {e}")
            return False

    def start_keep_alive(self):
        """keep-alive 명령을 주기적으로 전송하여 연결 유지"""
        self.keep_alive_active = True
        keep_alive_thread = threading.Thread(target=self.keep_alive_worker)
        keep_alive_thread.daemon = True
        keep_alive_thread.start()
        print("🔄 Keep-alive 모니터링 시작")

    def keep_alive_worker(self):
        """백그라운드에서 keep-alive 명령 전송 (2초마다 command 전송)"""
        while self.keep_alive_active:
            try:
                time.sleep(2)  # 2초마다 실행으로 변경
                if self.keep_alive_active:
                    # command 명령으로 연결 유지 (자동 착륙 방지)
                    try:
                        response = self.tello.send_command_without_return("command")
                        # 로그 출력하지 않음 (너무 많은 출력 방지)
                    except:
                        # command 실패시 배터리 상태 확인으로 대체
                        battery = self.tello.get_battery()
                        if battery <= 0:
                            print("⚠️ 드론 응답 없음 - 연결 상태 확인 필요")
            except Exception as e:
                if self.keep_alive_active:  # 활성 상태일 때만 에러 출력
                    print(f"⚠️ Keep-alive 오류: {e}")
                time.sleep(2)  # 오류 시 2초 대기

    def stop_keep_alive(self):
        """keep-alive 중지"""
        self.keep_alive_active = False
        print("🔄 Keep-alive 모니터링 중지")

    def start_video_stream(self):
        """비디오 스트리밍 시작"""
        try:
            self.tello.streamon()
            self.streaming = True
            print("📹 비디오 스트리밍 시작")
            
            # 프레임 캡처 스레드 시작
            frame_thread = threading.Thread(target=self.update_frame)
            frame_thread.daemon = True
            frame_thread.start()
            
            time.sleep(3)  # 스트리밍 안정화 대기
            
        except Exception as e:
            print(f"❌ 스트리밍 시작 실패: {e}")
            self.streaming = False
    
    def update_frame(self):
        """프레임 업데이트 (백그라운드 스레드)"""
        frame_error_count = 0

        while self.streaming:
            try:
                self.frame = self.tello.get_frame_read().frame
                frame_error_count = 0  # 성공시 에러 카운트 리셋

            except Exception as e:
                frame_error_count += 1

                if frame_error_count <= 1:
                    print(f"⚠️ 프레임 읽기 오류 ({frame_error_count}): {e}")

                elif frame_error_count == 10:
                    print("⚠️ 프레임 읽기 연속 실패 중... (로그 출력 제한)")

                time.sleep(0.1)  # 연속 실패시 잠시 대기
    
    def get_next_filename(self):
        """중복되지 않는 다음 파일명 생성"""
        # 기존 img_XXXX.jpg 패턴 파일들 검색
        pattern = os.path.join(self.save_path, "img_*.jpg")
        existing_files = glob.glob(pattern)

        # 기존 파일에서 숫자 추출
        existing_numbers = []

        for file_path in existing_files:
            filename = os.path.basename(file_path)
            match = re.match(r'img_(\d{4})\.jpg', filename)

            if match:
                existing_numbers.append(int(match.group(1)))

        # 다음 번호 결정 (1부터 시작)
        if existing_numbers:
            next_number = max(existing_numbers) + 1

        else:
            next_number = 1

        return f"img_{next_number:04d}.jpg"

    def capture_image(self, angle, index):
        """현재 프레임 캡처 및 RGB 형식으로 저장"""
        if self.frame is None:
            print(f"⚠️ 프레임 캡처 실패 (각도: {angle}°)")
            return False

        try:
            # 중복되지 않는 파일명 생성
            filename = self.get_next_filename()
            filepath = os.path.join(self.save_path, filename)

            # djitellopy의 프레임은 이미 RGB 형식이므로 변환 없이 직접 사용
            pil_image = Image.fromarray(self.frame.astype(np.uint8))
            pil_image.save(filepath, format='JPEG', quality=95)
            print(f"💾 RGB 이미지 저장: {filename}")

            # 디버그: 프레임 형식 확인
            print(f"   프레임 형태: {self.frame.shape}, 데이터 타입: {self.frame.dtype}")

            return True

        except Exception as e:
            print(f"❌ 이미지 저장 실패: {e}")
            return False
    
    
    def execute_mission(self):
        """전체 미션 실행"""
        print("\n🚁 === 드론 미션 시작 ===\n")
        
        try:
            # 1. 드론 연결
            if not self.connect_drone():
                return
            
            # 2. 이륙
            print("🛫 이륙 중...")
            self.tello.takeoff()
            time.sleep(1)
            
            # 3. 목표 고도로 상승 (5m) - 시간 기반 접근
            print(f"⬆️ 목표 고도 {self.target_height}cm로 상승 중...")

            # 이륙 후 초기 고도 확인
            time.sleep(2)  # 이륙 안정화 대기
            initial_height = self.tello.get_height()
            print(f"📏 이륙 후 고도: {initial_height}cm")

            # 시간 기반으로 단계적 상승 (센서가 정확하지 않을 수 있으므로)
            total_climb_steps = 1  # 10단계로 나누어 상승 (더 안전하게)
            step_distance = 500    # 각 단계마다 50cm (더 작은 단위)

            print(f"📐 {total_climb_steps}단계로 나누어 상승 (각 단계 {step_distance}cm)")

            for step in range(total_climb_steps):
                print(f"\n단계 {step + 1}/{total_climb_steps}: {step_distance}cm 상승")

                retry_count = 0
                max_retries = 3
                step_success = False

                while retry_count < max_retries and not step_success:
                    try:
                        print(f"   시도 {retry_count + 1}/{max_retries}...")

                        # 상승 명령 전송
                        result = self.tello.send_command_with_return(f"up {step_distance}")
                        print(f"   응답: {result}")

                        if result == "ok":
                            step_success = True
                            print(f"   ✅ {step_distance}cm 상승 완료")

                            # 충분한 안정화 시간
                            time.sleep(3)

                            # 고도 확인 (참고용)
                            try:
                                height = self.tello.get_height()
                                print(f"   📏 센서 고도: {height}cm")
                            except:
                                print(f"   📏 센서 고도: 읽기 실패")

                        else:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"   ⚠️ 재시도 필요... 2초 대기")
                                time.sleep(2)

                    except Exception as e:
                        retry_count += 1
                        print(f"   ❌ 오류: {e}")
                        if retry_count < max_retries:
                            time.sleep(2)

                if not step_success:
                    print(f"   ⚠️ 단계 {step + 1} 실패 - 다음 단계 계속")

            print(f"\n✅ 상승 완료! 예상 고도: 약 {total_climb_steps * step_distance}cm")

            # 최종 안정화
            time.sleep(3)
            try:
                final_height = self.tello.get_height()
                print(f"📏 최종 센서 고도: {final_height}cm")
            except:
                print(f"📏 최종 고도: 센서 읽기 실패 (예상: 약 500cm)")
            
            # 4. 비디오 스트리밍 시작
            self.start_video_stream()
            
            # 5. 360도 회전하며 촬영 (30도씩, 각 위치에서 5개씩)
            angles = list(range(0, 30, self.rotation_angle))  # 0, 30, 60, ..., 330도
            total_positions = len(angles)
            images_per_position = 75
            total_images = total_positions * images_per_position

            print(f"\n📸 총 {total_positions}개 위치에서 각각 {images_per_position}개씩 촬영 (총 {total_images}개)")
            print("=" * 60)

            for idx, angle in enumerate(angles, 1):
                print(f"\n[위치 {idx}/{total_positions}] 각도 {angle}° 위치")

                # 회전 (첫 번째 위치가 아닐 때만)
                if angle > 0:
                    print(f"↻ {self.rotation_angle}도 회전 중...")

                    # 회전 명령 재시도 로직
                    rotation_success = False
                    for attempt in range(3):  # 최대 3회 시도
                        try:
                            print(f"   회전 시도 {attempt + 1}/3...")
                            self.tello.rotate_clockwise(self.rotation_angle)
                            print(f"   회전 명령 전송 완료")
                            rotation_success = True
                            break

                        except Exception as e:
                            print(f"   회전 시도 {attempt + 1} 실패: {e}")
                            if attempt < 2:  # 마지막 시도가 아니면 잠시 대기
                                time.sleep(1)

                    if rotation_success:
                        time.sleep(3)  # 회전 안정화 시간 (3초)
                        print(f"   위치 안정화 중...")
                        time.sleep(1)
                    else:
                        print(f"⚠️ 회전 명령 최종 실패 - 현재 위치에서 촬영 계속")
                        time.sleep(2)

                # 현재 위치에서 5개 이미지 촬영
                print(f"⏱️ {images_per_position}개 이미지 촬영 중...")

                for capture_num in range(images_per_position):
                    time.sleep(1)  # 1초 간격
                    success = self.capture_image(angle, capture_num + 1)
                    if success:
                        print(f"   [{capture_num + 1}/{images_per_position}] 각도 {angle}° 캡처 완료")
                    else:
                        print(f"   [{capture_num + 1}/{images_per_position}] 각도 {angle}° 캡처 실패")

                # 각 위치마다 배터리 체크
                battery = self.tello.get_battery()
                print(f"🔋 배터리 상태: {battery}% (위치 {idx}/{total_positions} 완료)")

                if battery < 15:
                    print("⚠️ 배터리 위험 수준! 긴급 착륙")
                    break  # 현재 위치까지만 촬영하고 중단

                # 3번째 위치마다 추가 안정화 시간
                if idx % 3 == 0 and idx < total_positions:
                    print("   📍 위치 안정화 대기...")
                    time.sleep(1)
            
            print(f"\n✅ 360도 회전 촬영 완료! (총 {len(angles)}개 위치)")
            
        except Exception as e:
            print(f"❌ 미션 실행 중 오류: {e}")
            
        finally:
            # 6. 착륙 및 정리
            self.safe_landing()
    
    def safe_landing(self):
        """안전 착륙 및 리소스 정리"""
        try:
            print("\n🛬 착륙 준비...")

            # keep-alive 중지
            self.stop_keep_alive()

            # 스트리밍 종료
            if self.streaming:
                self.streaming = False
                self.tello.streamoff()

            # 착륙
            self.tello.land()
            time.sleep(3)

            # 연결 종료
            self.tello.end()

            print("✅ 안전하게 착륙 완료!")

        except Exception as e:
            print(f"⚠️ 착륙 중 오류: {e}")
            # 비상 착륙
            try:
                self.tello.emergency()

            except:
                pass

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("       Tello EDU 360° 회전 이미지 촬영 프로그램")
    print("=" * 60)
    
    # 드론 촬영 실행
    drone = TelloDroneCapture()
    
    try:
        drone.execute_mission()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단 - 긴급 착륙")
        drone.stop_keep_alive()
        drone.safe_landing()

    except Exception as e:
        print(f"\n❌ 예기치 않은 오류: {e}")
        drone.stop_keep_alive()
        drone.safe_landing()
    
    print("\n📊 === 미션 완료 ===")
    print(f"💾 이미지 저장 위치: {drone.save_path}")
    
    # 저장된 이미지 개수 확인
    if os.path.exists(drone.save_path):
        images = [f for f in os.listdir(drone.save_path) if f.endswith('.jpg')]
        print(f"📸 총 저장된 이미지: {len(images)}개")

if __name__ == "__main__":
    main()