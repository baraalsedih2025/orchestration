#!/usr/bin/env python3
"""
SLURM Multi-Server ML Training Demo
===================================

This script demonstrates distributed ML training across 4 Docker containers
using SLURM orchestration and PyTorch DDP. All parameters are loaded from config.json.

Configuration from config.json:
- 4 servers (Docker containers)
- 4 GPUs per server (16 total GPUs)
- SLURM job orchestration
- PyTorch DDP distributed training

Author: ML Team
Date: 2025
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Server-%(process)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class SLURMMultiServerTrainer:
    """
    Multi-server distributed training using SLURM orchestration
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # SLURM environment variables
        self.rank = int(os.environ.get('SLURM_PROCID', 0))
        self.local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        self.node_id = int(os.environ.get('SLURM_NODEID', 0))
        
        # SLURM job information
        self.job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        self.job_name = os.environ.get('SLURM_JOB_NAME', 'ml_training')
        self.nodes = os.environ.get('SLURM_JOB_NODELIST', 'unknown')
        
        # Master address and port
        self.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.master_port = os.environ.get('MASTER_PORT', '12355')
        
        logger.info(f"Initializing SLURM trainer - Rank: {self.rank}, Local Rank: {self.local_rank}")
        logger.info(f"World Size: {self.world_size}, Node ID: {self.node_id}")
        logger.info(f"SLURM Job ID: {self.job_id}, Nodes: {self.nodes}")
        logger.info(f"Master: {self.master_addr}:{self.master_port}")
        
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        
    def setup_distributed(self):
        """Initialize distributed training using config.json backend"""
        backend = self.config.get('distributed', {}).get('backend', 'nccl')
        
        # Use gloo backend if no CUDA available
        if not torch.cuda.is_available() and backend == 'nccl':
            backend = 'gloo'
            logger.info(f"⚠️ CUDA not available, switching from nccl to {backend}")
        
        try:
            dist.init_process_group(
                backend=backend,
                init_method=f'tcp://{self.master_addr}:{self.master_port}',
                world_size=self.world_size,
                rank=self.rank
            )
            logger.info(f"✓ Distributed training initialized on rank {self.rank} using {backend}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize distributed training: {e}")
            sys.exit(1)
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            logger.info(f"✓ Using CUDA device {self.local_rank}")
        else:
            logger.info("⚠ CUDA not available, using CPU")
    
    def setup_model(self):
        """Setup the neural network model using config.json parameters"""
        model_config = self.config.get('training', {}).get('model', {})
        
        # Use ResNet50 as specified in config
        if model_config.get('name') == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=model_config.get('pretrained', False))
            num_classes = model_config.get('num_classes', 10)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            # Fallback to SimpleCNN
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes=10):
                    super(SimpleCNN, self).__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(128 * 4 * 4, 512)
                    self.fc2 = nn.Linear(512, num_classes)
                    self.dropout = nn.Dropout(0.5)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = self.pool(self.relu(self.conv3(x)))
                    x = x.view(-1, 128 * 4 * 4)
                    x = self.dropout(self.relu(self.fc1(x)))
                    x = self.fc2(x)
                    return x
            
            self.model = SimpleCNN(num_classes=model_config.get('num_classes', 10))
        
        # Move model to GPU and wrap with DDP
        if torch.cuda.is_available():
            # Move to the correct GPU device
            device = torch.device(f'cuda:{self.local_rank}')
            self.model = self.model.to(device)
            # Wrap with DDP
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            # CPU mode - still wrap with DDP for distributed training
            self.model = DDP(self.model)
        
        # Setup optimizer using config.json
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'sgd')
        
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('learning_rate', 0.1),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('learning_rate', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"✓ Model setup complete on rank {self.rank} - {model_config.get('name', 'SimpleCNN')}")
    
    def setup_data(self):
        """Setup data loaders using config.json parameters"""
        data_config = self.config.get('training', {}).get('data', {})
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224) if data_config.get('dataset') == 'imagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if data_config.get('dataset') == 'imagenet' else [0.4914, 0.4822, 0.4465],
                std=[0.229, 0.224, 0.225] if data_config.get('dataset') == 'imagenet' else [0.2023, 0.1994, 0.2010]
            )
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256) if data_config.get('dataset') == 'imagenet' else transforms.Resize(32),
            transforms.CenterCrop(224) if data_config.get('dataset') == 'imagenet' else transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if data_config.get('dataset') == 'imagenet' else [0.4914, 0.4822, 0.4465],
                std=[0.229, 0.224, 0.225] if data_config.get('dataset') == 'imagenet' else [0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Load datasets
        dataset_name = data_config.get('dataset', 'cifar10')
        data_path = data_config.get('data_path', './data')
        
        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=True,
                download=True,
                transform=transform_train
            )
            val_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=False,
                download=True,
                transform=transform_val
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create data loaders using config.json parameters
        batch_size = data_config.get('batch_size', 32)
        num_workers = data_config.get('num_workers', 4)
        pin_memory = data_config.get('pin_memory', True)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        
        logger.info(f"✓ Data loaders setup complete on rank {self.rank}")
        logger.info(f"  Dataset: {dataset_name}, Batch size: {batch_size}, Workers: {num_workers}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to the correct device
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{self.local_rank}')
                data, target = data.to(device), target.to(device)
            else:
                data, target = data, target
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0 and self.rank == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Acc: {100.*correct/total:.2f}%')
        
        # Synchronize metrics across all processes
        if torch.cuda.is_available():
            metrics = torch.tensor([total_loss, correct, total]).cuda()
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        else:
            metrics = torch.tensor([total_loss, correct, total])
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics[0].item() / len(self.train_loader)
        accuracy = 100. * metrics[1].item() / metrics[2].item()
        
        if self.rank == 0:
            logger.info(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}, '
                       f'Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                # Move data to the correct device
                if torch.cuda.is_available():
                    device = torch.device(f'cuda:{self.local_rank}')
                    data, target = data.to(device), target.to(device)
                else:
                    data, target = data, target
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # Synchronize metrics
        if torch.cuda.is_available():
            metrics = torch.tensor([total_loss, correct, total]).cuda()
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        else:
            metrics = torch.tensor([total_loss, correct, total])
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics[0].item() / len(self.val_loader)
        accuracy = 100. * metrics[1].item() / metrics[2].item()
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, best_acc: float):
        """Save model checkpoint using config.json settings"""
        checkpoint_config = self.config.get('checkpointing', {})
        
        if self.rank == 0:  # Only save on rank 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': best_acc,
                'config': self.config
            }
            
            checkpoint_dir = checkpoint_config.get('checkpoint_dir', './checkpoints')
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_file)
            
            logger.info(f'✓ Checkpoint saved: {checkpoint_file}')
    
    def train(self):
        """Main training loop using config.json parameters"""
        training_config = self.config.get('training', {}).get('training', {})
        num_epochs = training_config.get('num_epochs', 100)
        save_frequency = training_config.get('save_frequency', 10)
        best_acc = 0.0
        
        if self.rank == 0:
            logger.info(f"🚀 Starting SLURM distributed training")
            logger.info(f"   World size: {self.world_size} (4 GPUs total), Epochs: {num_epochs}")
            logger.info(f"   Save frequency: {save_frequency}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - start_time
            
            if self.rank == 0:
                logger.info(f'📊 Epoch {epoch} - Time: {epoch_time:.2f}s - '
                           f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
                
                # Save checkpoint based on frequency or best accuracy
                if epoch % save_frequency == 0 or val_acc > best_acc:
                    best_acc = max(best_acc, val_acc)
                    self.save_checkpoint(epoch, best_acc)
        
        if self.rank == 0:
            logger.info(f'✅ SLURM training completed. Best validation accuracy: {best_acc:.2f}%')
    
    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()
        logger.info(f"✓ SLURM trainer cleanup completed on rank {self.rank}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SLURM Multi-Server ML Training Demo')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--rank', type=int, default=0,
                       help='Process rank')
    parser.add_argument('--world-size', type=int, default=16,
                       help='World size')
    
    args = parser.parse_args()
    
    # Load configuration from config.json
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    logger.info(f"✓ Loaded configuration from {args.config}")
    
    # Create necessary directories
    data_path = config.get('training', {}).get('data', {}).get('data_path', './data')
    checkpoint_dir = config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints')
    log_dir = config.get('logging', {}).get('log_dir', './logs')
    
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SLURMMultiServerTrainer(config)
    
    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == '__main__':
    main()
