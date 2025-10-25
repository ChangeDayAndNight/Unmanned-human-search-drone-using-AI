import os
import re

def rename_image_files():
    """
    현재 디렉토리의 img_xxxx.jpg 형식 파일들을 img_001.jpg, img_002.jpg 순으로 이름을 변경합니다.
    """
    # 현재 디렉토리 경로
    current_dir = os.getcwd()

    # img_로 시작하고 .jpg로 끝나는 파일들을 찾기
    pattern = re.compile(r'^img_\d+\.jpg$')
    image_files = []

    for filename in os.listdir(current_dir):
        if pattern.match(filename):
            image_files.append(filename)

    # 파일명을 숫자순으로 정렬
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    print("파일 이름 변경을 시작합니다...")

    # 임시 디렉토리 생성하여 충돌 방지
    temp_dir = os.path.join(current_dir, "temp_rename")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1단계: 모든 파일을 임시 디렉토리로 이동
        for i, old_filename in enumerate(image_files, 1):
            old_path = os.path.join(current_dir, old_filename)
            temp_path = os.path.join(temp_dir, f"temp_{i:03d}.jpg")
            os.rename(old_path, temp_path)
            print(f"임시 이동: {old_filename} -> temp_{i:03d}.jpg")

        # 2단계: 임시 파일들을 새로운 이름으로 원래 디렉토리로 이동
        for i in range(1, len(image_files) + 1):
            temp_path = os.path.join(temp_dir, f"temp_{i:03d}.jpg")
            new_filename = f"img_{i:03d}.jpg"
            new_path = os.path.join(current_dir, new_filename)
            os.rename(temp_path, new_path)
            print(f"최종 이름 변경: temp_{i:03d}.jpg -> {new_filename}")

        # 임시 디렉토리 삭제
        os.rmdir(temp_dir)

        print(f"\n완료! {len(image_files)}개의 파일이 img_001.jpg부터 img_{len(image_files):03d}.jpg로 이름이 변경되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        # 오류 발생 시 임시 파일들을 원래 이름으로 복구 시도
        print("파일 복구를 시도합니다...")
        for i, old_filename in enumerate(image_files, 1):
            temp_path = os.path.join(temp_dir, f"temp_{i:03d}.jpg")
            if os.path.exists(temp_path):
                original_path = os.path.join(current_dir, old_filename)
                os.rename(temp_path, original_path)
                print(f"복구: temp_{i:03d}.jpg -> {old_filename}")

        # 임시 디렉토리가 비어있다면 삭제
        try:
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    # 사용자 확인
    response = input("현재 디렉토리의 모든 img_xxxx.jpg 파일들을 img_001.jpg부터 순차적으로 이름을 변경하시겠습니까? (y/n): ")

    if response.lower() in ['y', 'yes']:
        rename_image_files()
    else:
        print("작업이 취소되었습니다.")