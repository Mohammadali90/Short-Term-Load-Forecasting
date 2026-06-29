#!/usr/bin/env python
# coding: utf-8

"""
Run all models and experiments in the project.
"""

print("=" * 70)
print("POWER CONSUMPTION FORECASTING - ALL EXPERIMENTS")
print("=" * 70)

try:
    print("\n1. Running Multi-Seed XGBoost...")
    import multi_seed_xgboost
    multi_seed_xgboost.run_multi_seed_xgboost()
except Exception as e:
    print(f"XGBoost failed: {e}")

try:
    print("\n2. Running Multi-Seed Prophet...")
    import multi_seed_prophet
    multi_seed_prophet.run_multi_seed_prophet()
except Exception as e:
    print(f"Prophet failed: {e}")

try:
    print("\n3. Running TE-BiLSTM...")
    import run_te_bilstm
    run_te_bilstm.main()
except Exception as e:
    print(f"TE-BiLSTM failed: {e}")

try:
    print("\n4. Running Statistical Analysis...")
    import statistics
    statistics.run_statistical_analysis()
except Exception as e:
    print(f"Statistics failed: {e}")

print("\n🎉 All experiments completed!")
print("Check the generated .docx and .csv files for results.")