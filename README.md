# Egocentric Vision-Language-Action (VLA) Pipeline

This repository contains a complete, end-to-end pipeline for fine-tuning Vision-Language-Action (VLA) models (specifically SmolVLA) on custom egocentric manipulation datasets.

## Pipeline Overview

1. **Feature Extraction (`1_extract_features.py`)**: Processes raw egocentric videos and extracts relevant states and actions.
2. **Dataset Generation (`4_build_lerobot_dataset.py`)**: Converts the extracted features into a strict Parquet-based schema compatible with Hugging Face's LeRobot v2.0/v3.0 dataset standards.
3. **Model & Colab Fine-tuning (`notebooks/colab_finetune_smolvla.py`)**: Contains the environment setup, dataset patching, and Hugging Face `lerobot-train` commands necessary to fine-tune a 450M parameter SmolVLA model on Google Colab (T4 GPU).
4. **Evaluation**: Includes scripts and JSON logs (`eval_results.json`, `training_history.json`) to track position MSE and gripper accuracy.

## Recent Updates & Fixes
- Implemented robust LeRobot schema alignment (handling `dataset_from_index`, `dataset_to_index`, and `from_timestamp`).
- Resolved multi-dataset meta indexing errors.
- Structured the pipeline to allow seamless uploading of local datasets to Google Drive for Colab-based GPU training.

## Next Steps
- Re-run full 10,000 step training on Colab with persistent Google Drive output directories.
- Implement live hardware inference.
