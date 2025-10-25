import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 경로 (사용자 파일명으로 수정)
file_path = 'results.xlsx'
df = pd.read_excel(file_path)

# epoch 컬럼 확인
if 'epoch' not in df.columns:
    raise ValueError("엑셀 파일에 'epoch' 열이 포함되어야 합니다.")

# 시각화할 메트릭 리스트 (epoch 제외)
metrics = [col for col in df.columns if col != 'epoch']

# 그래프 스타일
plt.style.use('seaborn-v0_8-darkgrid')

# 각 메트릭마다 독립된 그래프 생성
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df[metric], marker='o', linewidth=2)
    plt.title(f'{metric} per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
