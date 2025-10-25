"""
Simple Autonomous Flight Launcher
간단한 자율 비행 실행기
"""

import os
import glob
import sys
from datetime import datetime

def find_latest_model():
    """최신 훈련된 모델 찾기"""
    model_dir = "models"
    if not os.path.exists(model_dir):
        return None

    # Inference 모델 우선 검색
    inference_models = glob.glob(f"{model_dir}/DQN_Inference_*.pth")
    if inference_models:
        latest_inference = max(inference_models, key=os.path.getctime)
        return latest_inference

    # Final 모델 검색
    final_models = glob.glob(f"{model_dir}/DQN_Final_Model_*.pth")
    if final_models:
        latest_final = max(final_models, key=os.path.getctime)
        return latest_final

    return None

def main():
    print("🚁 DQN Autonomous Flight Launcher")
    print("=" * 50)

    # 최신 모델 찾기
    latest_model = find_latest_model()

    if latest_model is None:
        print("❌ No trained model found!")
        print("   Please run train_fast_500.py first to train a model.")
        return

    print(f"✅ Found trained model: {latest_model}")

    # 모델 정보 확인
    try:
        import torch
        model_data = torch.load(latest_model, map_location='cpu')
        if 'save_timestamp' in model_data:
            print(f"📅 Model trained: {model_data['save_timestamp']}")
        if 'training_episodes' in model_data:
            print(f"📈 Training episodes: {model_data['training_episodes']}")
        if 'model_type' in model_data:
            print(f"🤖 Model type: {model_data['model_type']}")
    except Exception as e:
        print(f"⚠️ Could not read model info: {e}")

    print("\n🎯 Flight Options:")
    print("1. Single target mission (recommended)")
    print("2. Multi-target mission")
    print("3. Continuous flight mode")
    print("4. Custom parameters")

    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == "1":
                # Single target mission
                print("\n🚀 Starting single target mission...")
                os.system(f'python autonomous_flight.py --model "{latest_model}" --target-mode single --max-steps 500')
                break

            elif choice == "2":
                # Multi-target mission
                print("\n🚀 Starting multi-target mission...")
                os.system(f'python autonomous_flight.py --model "{latest_model}" --target-mode multi --max-steps 1000')
                break

            elif choice == "3":
                # Continuous mode
                print("\n🚀 Starting continuous flight mode...")
                os.system(f'python autonomous_flight.py --model "{latest_model}" --target-mode single --continuous --max-steps 500')
                break

            elif choice == "4":
                # Custom parameters
                print("\n⚙️ Custom parameters:")
                target_mode = input("Target mode (single/multi) [single]: ").strip() or "single"
                max_steps = input("Max steps [500]: ").strip() or "500"
                continuous = input("Continuous mode? (y/N): ").strip().lower() == 'y'

                cmd = f'python autonomous_flight.py --model "{latest_model}" --target-mode {target_mode} --max-steps {max_steps}'
                if continuous:
                    cmd += " --continuous"

                print(f"\n🚀 Starting flight with custom parameters...")
                os.system(cmd)
                break

            else:
                print("❌ Invalid choice. Please select 1-4.")

        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()