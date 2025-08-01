#!/bin/bash
# Dondurma kalite kontrol sistemi çalıştırma örneği

# Temel kullanım
python ice_cream_quality_control_v3.py \
    --good_dir "good_ice_creams" \
    --test_dir "production_batch_20250801" \
    --anomaly_folder "detected_anomalies" \
    --confidence 0.95

# Gelişmiş ensemble kullanımı
python ice_cream_quality_control_v3.py \
    --good_dir "reference_ice_creams" \
    --test_dir "daily_production" \
    --anomaly_folder "quality_check_results" \
    --architectures resnet18 resnet50 resnet101 \
    --tiling_strategy multi_scale \
    --ensemble_method weighted_avg \
    --confidence 0.98 \
    --use_adaptive_roi \
    --save_reports \
    --device cuda

# Hızlı tarama modu (sadece ResNet18)
python ice_cream_quality_control_v3.py \
    --good_dir "good_samples" \
    --test_dir "quick_check" \
    --anomaly_folder "quick_results" \
    --architectures resnet18 \
    --tiling_strategy center_focused \
    --manual_threshold 0.15 \
    --device cuda
