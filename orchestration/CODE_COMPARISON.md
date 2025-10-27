# Training Code Comparison: test_single vs Docker

## Key Point: **Same Training Code, Different Environment**

Both `test_single.sh` and Docker setup use **the exact same ML training code**: `slurm_training_demo.py`

## Differences

### Environment Variables

**test_single.sh (Local)**:
```bash
WORLD_SIZE=1    # Single worker
RANK=0         # Only rank 0
SLURM_NTASKS=1 # 1 task
```

**Docker (Distributed)**:
```bash
WORLD_SIZE=4         # 4 workers
RANK=0,1,2,3        # Different rank per container
SLURM_NTASKS=4      # 4 tasks total
```

## Training Code Behavior

The `slurm_training_demo.py` script reads these environment variables:

```python
# From slurm_training_demo.py
self.rank = int(os.environ.get('SLURM_PROCID', 0))
self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
```

### When WORLD_SIZE=1 (test_single.sh):
- Runs as single process
- No DDP synchronization needed
- All data processed by one worker

### When WORLD_SIZE=4 (Docker):
- Runs as 4 processes (one per container)
- DDP synchronizes gradients across all 4
- Data distributed across 4 workers
- Model updates averaged across workers

## What Changes Automatically

The training code automatically adapts:

1. **Distributed Sampler**: 
   - WORLD_SIZE=1: Gets all data
   - WORLD_SIZE=4: Gets 1/4 of data

2. **Gradient Synchronization**:
   - WORLD_SIZE=1: No sync needed
   - WORLD_SIZE=4: `dist.all_reduce()` across all workers

3. **Checkpointing**:
   - WORLD_SIZE=1: Always saves
   - WORLD_SIZE=4: Only rank 0 saves

4. **Logging**:
   - Both: Only rank 0 logs to console

## Summary

**Same code, different scale:**

- **test_single.sh** → WORLD_SIZE=1 → Single worker → Sequential training
- **Docker setup** → WORLD_SIZE=4 → 4 workers → Distributed training with DDP

The code is smart enough to handle both modes automatically!

