# Power Consumption Forecasting using Deep Learning

This repository contains the official implementation of the paper on **hourly power consumption forecasting** using classical and deep learning models, with a focus on **TE-BiLSTM**.

## Models
- XGBoost
- Prophet
- LSTM / BiLSTM
- CNN-LSTM
- TCN
- Transformer
- **TE-BiLSTM** (Proposed best model)

## Features
- Multi-seed experiments for robustness
- Comprehensive evaluation (MAE, RMSE, R², MAPE)
- Peak demand analysis
- Robustness analysis (season, weekday/weekend, peak/off-peak)
- Statistical significance tests (T-test)
- Automatic Word report generation

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo>
cd power-forecasting
pip install -r requirements.txt

# 2. Run main model
python run_te_bilstm.py

# 3. Run all experiments
python run_all.py