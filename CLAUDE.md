# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status
DQN AirSim 무인 인명 탐색 드론 프로젝트 - 완전 구현됨

### Development Commands
- Start training: `python train.py`
- Resume training: `python train.py --resume checkpoints/latest_model.pth`
- Run autonomous flight: `python autonomous_flight.py --model models/DQN_Final_model.pth`
- Test curriculum: `python curriculum.py`

### Architecture Overview
Advanced DQN implementation with:
- Double DQN + Dueling DQN + Prioritized Experience Replay
- Multi-step Learning (3-step)
- Mixed Precision Training (RTX 3060Ti optimized)
- Self-Attention mechanism
- CNN + MLP hybrid architecture for image + position fusion

### Project Structure
```
DQN_No_Human/
├── config.py                     # Configuration and hyperparameters
├── curriculum.py                 # Curriculum learning configuration
├── environment.py                # AirSim environment wrapper
├── network.py                    # DQN network architectures
├── dqn_agent.py                  # DQN agent with latest techniques
├── utils.py                      # Logging, visualization, monitoring
├── train.py                      # Main training loop
├── autonomous_flight.py          # Autonomous flight with trained model
├── real_time_monitor.py          # Real-time training monitoring
├── performance_analyzer.py       # Performance analysis and Excel reports
├── optimized_memory_manager.py   # Memory optimization utilities
├── run_autonomous.py             # Autonomous flight runner
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

### Important Notes
- **Hardware Optimized**: RTX 3060Ti (8GB VRAM) + 16GB RAM + Ryzen 5 5600X
- **AirSim Required**: Must have AirSim 1.8.1 + Unreal Engine 4.27 running
- **Python 3.10**: Tested with Python 3.10 environment
- **GPU Memory**: Uses mixed precision training for 8GB VRAM efficiency
- **Batch Size**: Optimized at 32 for RTX 3060Ti
- **Random Seed**: Set to 42 for reproducibility

### Quick Start
1. Install requirements: `pip install -r requirements.txt`
2. Setup AirSim with settings.json configuration
3. Test curriculum: `python curriculum.py`
4. Start training: `python train.py`
5. Run autonomous flight: `python autonomous_flight.py --model models/trained_model.pth`