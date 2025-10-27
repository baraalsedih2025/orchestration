#!/usr/bin/env python3
"""
Simple single-worker demo of the training code
"""

import os
import sys
import json
import torch

# Set up environment for single worker
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['SLURM_PROCID'] = '0'
os.environ['SLURM_LOCALID'] = '0'
os.environ['SLURM_NTASKS'] = '1'
os.environ['SLURM_NODEID'] = '0'
os.environ['SLURM_JOB_ID'] = 'local-demo-001'
os.environ['SLURM_JOB_NAME'] = 'ml_training_demo_local'
os.environ['SLURM_JOB_NODELIST'] = 'localhost'

print("="*60)
print("Distributed ML Training Demo - Local Run")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Modify config for quick demo (fewer epochs)
config['training']['training']['num_epochs'] = 2
config['training']['data']['batch_size'] = 16

print("Configuration loaded:")
print(f"  Model: {config['training']['model']['name']}")
print(f"  Dataset: {config['training']['data']['dataset']}")
print(f"  Batch Size: {config['training']['data']['batch_size']}")
print(f"  Epochs: {config['training']['training']['num_epochs']}")
print()

# Check if we need to initialize distributed training
# For single worker, we'll use a simplified approach
print("Starting training...")
print("Note: This is a single-worker demo (not true distributed training)")
print("For full 4-worker distributed training, run with Docker on your machine")
print()

try:
    # Try to import and run the training
    import slurm_training_demo
    
    trainer = slurm_training_demo.SLURMMultiServerTrainer(config)
    
    print("\n🚀 Starting training...")
    trainer.train()
    trainer.cleanup()
    
    print("\n✅ Training completed!")
    print("Checkpoints saved in: ./checkpoints/")
    print("Logs saved in: ./logs/")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Demo complete!")
print("To run the full 4-worker distributed training:")
print("  ./run_training_docker.sh")
print("  (Requires Docker and 4 GPUs)")
print("="*60)

