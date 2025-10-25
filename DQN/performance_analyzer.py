"""
DQN 성능 분석 및 Excel 파일 생성
500 에피소드 학습 결과를 종합적으로 분석하여 xlsx 파일로 저장
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any
import json

class PerformanceAnalyzer:
    """DQN 학습 성능 분석 및 Excel 보고서 생성"""

    def __init__(self):
        self.episode_data = []
        self.stage_statistics = {}

    def add_episode_data(self, episode: int, data: Dict[str, Any]):
        """Add episode data"""
        episode_info = {
            'Episode': episode,
            'Reward': data.get('reward', 0),
            'Episode_Length': data.get('length', 0),
            'Loss': data.get('loss', 0),
            'Targets_Found': data.get('targets_found', 0),
            'Epsilon': data.get('epsilon', 0),
            'Time_Seconds': data.get('time', 0),
            'Exploration_Coverage': data.get('exploration_coverage', 0),
            'Stage': data.get('stage', 'unknown'),
            'Learning_Rate': data.get('learning_rate', 0),
            'Collision': data.get('collision', False),
            'Success_Rate': 1 if data.get('targets_found', 0) > 0 else 0,
            'Efficiency': data.get('targets_found', 0) / max(data.get('length', 1), 1),
            'Termination_Reason': data.get('reason', 'unknown')
        }
        self.episode_data.append(episode_info)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        if not self.episode_data:
            return {}

        df = pd.DataFrame(self.episode_data)

        stats = {
            'Total_Episodes': len(df),
            'Average_Reward': df['Reward'].mean(),
            'Max_Reward': df['Reward'].max(),
            'Min_Reward': df['Reward'].min(),
            'Reward_Standard_Deviation': df['Reward'].std(),
            'Average_Episode_Length': df['Episode_Length'].mean(),
            'Average_Loss': df['Loss'].mean(),
            'Total_Targets_Found': df['Targets_Found'].sum(),
            'Average_Targets_Per_Episode': df['Targets_Found'].mean(),
            'Success_Rate_Percent': df['Success_Rate'].mean() * 100,
            'Average_Exploration_Coverage': df['Exploration_Coverage'].mean(),
            'Average_Efficiency': df['Efficiency'].mean(),
            'Total_Training_Time_Minutes': df['Time_Seconds'].sum() / 60,
            'Collision_Rate_Percent': (df['Collision'].sum() / len(df)) * 100,
            'Final_Epsilon': df['Epsilon'].iloc[-1] if len(df) > 0 else 0
        }

        return stats

    def calculate_stage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate stage-wise statistics"""
        if not self.episode_data:
            return {}

        df = pd.DataFrame(self.episode_data)
        stage_stats = {}

        for stage in df['Stage'].unique():
            stage_df = df[df['Stage'] == stage]
            stage_stats[stage] = {
                'Total_Episodes': len(stage_df),
                'Episode_Range': f"{stage_df['Episode'].min()}-{stage_df['Episode'].max()}",
                'Average_Reward': stage_df['Reward'].mean(),
                'Max_Reward': stage_df['Reward'].max(),
                'Min_Reward': stage_df['Reward'].min(),
                'Average_Episode_Length': stage_df['Episode_Length'].mean(),
                'Average_Loss': stage_df['Loss'].mean(),
                'Success_Rate_Percent': stage_df['Success_Rate'].mean() * 100,
                'Average_Targets_Found': stage_df['Targets_Found'].mean(),
                'Collision_Rate_Percent': (stage_df['Collision'].sum() / len(stage_df)) * 100,
                'Average_Efficiency': stage_df['Efficiency'].mean(),
                'Performance_Improvement': stage_df['Reward'].iloc[-10:].mean() - stage_df['Reward'].iloc[:10].mean() if len(stage_df) >= 20 else 0
            }

        return stage_stats

    def calculate_learning_progress(self) -> pd.DataFrame:
        """Calculate learning progress (10-episode moving average)"""
        if not self.episode_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.episode_data)

        # Calculate 10-episode moving average
        window = 10
        progress_data = []

        for i in range(window, len(df) + 1):
            window_data = df.iloc[i-window:i]
            progress_data.append({
                'Episode_Range': f"{i-window+1}-{i}",
                'Average_Reward': window_data['Reward'].mean(),
                'Average_Episode_Length': window_data['Episode_Length'].mean(),
                'Average_Loss': window_data['Loss'].mean(),
                'Success_Rate_Percent': window_data['Success_Rate'].mean() * 100,
                'Average_Targets_Found': window_data['Targets_Found'].mean(),
                'Collision_Rate_Percent': (window_data['Collision'].sum() / len(window_data)) * 100,
                'Average_Efficiency': window_data['Efficiency'].mean()
            })

        return pd.DataFrame(progress_data)

    def generate_excel_report(self, filename: str = None) -> str:
        """Excel 보고서 생성"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"DQN_Performance_Report_{timestamp}.xlsx"

        filepath = os.path.join("reports", filename)
        os.makedirs("reports", exist_ok=True)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. 에피소드별 상세 데이터
            episode_df = pd.DataFrame(self.episode_data)
            episode_df.to_excel(writer, sheet_name='Episode_Details', index=False)

            # 2. 전체 통계
            overall_stats = self.calculate_statistics()
            stats_df = pd.DataFrame(list(overall_stats.items()), columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)

            # 3. 스테이지별 통계
            stage_stats = self.calculate_stage_statistics()
            if stage_stats:
                stage_df = pd.DataFrame(stage_stats).T
                stage_df.to_excel(writer, sheet_name='Stage_Statistics')

            # 4. 학습 진행률 (이동평균)
            progress_df = self.calculate_learning_progress()
            if not progress_df.empty:
                progress_df.to_excel(writer, sheet_name='Learning_Progress', index=False)

            # 5. Summary Dashboard
            summary_data = {
                'Training_Summary': [
                    ['Total Episodes', overall_stats.get('Total_Episodes', 0)],
                    ['Final Average Reward', f"{overall_stats.get('Average_Reward', 0):.2f}"],
                    ['Best Episode Reward', f"{overall_stats.get('Max_Reward', 0):.2f}"],
                    ['Success Rate (%)', f"{overall_stats.get('Success_Rate_Percent', 0):.1f}%"],
                    ['Collision Rate (%)', f"{overall_stats.get('Collision_Rate_Percent', 0):.1f}%"],
                    ['Total Training Time (min)', f"{overall_stats.get('Total_Training_Time_Minutes', 0):.1f}"],
                    ['Average Efficiency', f"{overall_stats.get('Average_Efficiency', 0):.3f}"],
                    ['Final Epsilon', f"{overall_stats.get('Final_Epsilon', 0):.3f}"]
                ]
            }

            summary_df = pd.DataFrame(summary_data['Training_Summary'], columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary_Dashboard', index=False)

        print(f"📊 성능 보고서가 생성되었습니다: {filepath}")
        return filepath

    def save_raw_data(self, filename: str = None):
        """원시 데이터 JSON 형태로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_data_{timestamp}.json"

        filepath = os.path.join("reports", filename)
        os.makedirs("reports", exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.episode_data, f, indent=2, ensure_ascii=False)

        print(f"💾 원시 데이터가 저장되었습니다: {filepath}")
        return filepath

    def print_summary(self):
        """Print summary information to console"""
        stats = self.calculate_statistics()

        print("\n" + "="*80)
        print("🏆 DQN Training Performance Summary (500 Episodes)")
        print("="*80)

        print(f"📊 Total Episodes: {stats.get('Total_Episodes', 0)}")
        print(f"🎯 Average Reward: {stats.get('Average_Reward', 0):.2f}")
        print(f"🏅 Best Reward: {stats.get('Max_Reward', 0):.2f}")
        print(f"📈 Success Rate: {stats.get('Success_Rate_Percent', 0):.1f}%")
        print(f"💥 Collision Rate: {stats.get('Collision_Rate_Percent', 0):.1f}%")
        print(f"⏱️ Total Training Time: {stats.get('Total_Training_Time_Minutes', 0):.1f} min")
        print(f"🎲 Final Epsilon: {stats.get('Final_Epsilon', 0):.3f}")
        print(f"⚡ Average Efficiency: {stats.get('Average_Efficiency', 0):.3f}")

        print("\n📋 Stage Performance:")
        stage_stats = self.calculate_stage_statistics()
        for stage, stats_dict in stage_stats.items():
            print(f"  {stage}: Reward {stats_dict['Average_Reward']:.1f} | "
                  f"Success {stats_dict['Success_Rate_Percent']:.1f}% | "
                  f"Collision {stats_dict['Collision_Rate_Percent']:.1f}%")

        print("="*80)

# 전역 인스턴스
performance_analyzer = PerformanceAnalyzer()

if __name__ == "__main__":
    # 테스트용 샘플 데이터
    analyzer = PerformanceAnalyzer()

    # 샘플 데이터 추가
    for i in range(1, 11):
        sample_data = {
            'reward': np.random.normal(50, 20),
            'length': np.random.randint(100, 300),
            'loss': np.random.exponential(0.1),
            'targets_found': np.random.randint(0, 3),
            'epsilon': 1.0 - (i * 0.1),
            'time': np.random.uniform(30, 120),
            'exploration_coverage': np.random.uniform(0.3, 0.8),
            'stage': 'test_stage',
            'learning_rate': 0.001,
            'collision': np.random.choice([True, False], p=[0.2, 0.8]),
            'reason': 'test'
        }
        analyzer.add_episode_data(i, sample_data)

    # 보고서 생성 테스트
    analyzer.print_summary()
    report_path = analyzer.generate_excel_report("test_report.xlsx")
    print(f"테스트 보고서 생성 완료: {report_path}")