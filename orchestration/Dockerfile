FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Default command
CMD ["python", "slurm_training_demo.py"]
