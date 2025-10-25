from djitellopy import Tello
import time

class TelloDroneBatteryCheck:
    def __init__(self):
        self.tello = Tello()

    def connect_drone(self):
        """드론에 연결"""
        try:
            print("드론에 연결 중...")
            self.tello.connect()
            print("드론 연결 성공!")
            return True
        except Exception as e:
            print(f"드론 연결 실패: {e}")
            return False

    def BatteryCheck(self):
        """배터리 상태 확인"""
        try:
            battery_level = self.tello.get_battery()
            print(f"배터리 잔량: {battery_level}%")
            return battery_level
        
        except Exception as e:
            print(f"배터리 정보를 가져올 수 없습니다: {e}")
            return None

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("       Tello EDU Battery Check")
    print("=" * 60)

    drone = TelloDroneBatteryCheck()

    # 드론 연결
    if drone.connect_drone():
        # 잠시 대기 후 배터리 확인
        time.sleep(1)
        drone.BatteryCheck()

if __name__ == "__main__":
    main()