# RLGBS: Reinforcement Learning-Guided Beam Search

This repository contains the supplementary materials and source code for the paper "RLGBS: Reinforcement Learning-Guided Beam Search for process optimization in a paper machine dryer section".

## Citation

If you use this code in your research, please cite:

```bibtex
@article{PLACEHOLDER_2025,
  title={RLGBS: Reinforcement Learning-Guided Beam Search for process optimization in a paper machine dryer section},
  author={PLACEHOLDER},
  journal={PLACEHOLDER JOURNAL},
  year={2025},
  doi={PLACEHOLDER DOI}
}
```

## Overview

This project combines Reinforcement Learning ([PPO](https://arxiv.org/abs/1707.06347)) with beam search algorithms to optimize multi-cylinder paper dryer section temperature control in paper manufacturing processes. The implementation includes:

- A [gymnasium](https://arxiv.org/abs/2407.17032) environment wrapper for the C++ drum dryer numerical simulation program
- Modified [HuggingFace Transformers](https://github.com/huggingface/transformers) generation module adapted for RL beam search
- Training and evaluation scripts for reproducing paper results
- Baseline implementations for comparison

## System Requirements

- Ubuntu 24.04 or other compatible Linux distribution with Docker support
- ≥64GB system RAM (recommended for full batch evaluation)
- NVIDIA GPU (recommended for full batch evaluation)

## Setup

This section provides instructions for setting up the environment and reproducing the experimental results reported in the paper.
### 1. Create ramdisk

Create a ramdisk for frequently accessed simulation data:

```bash
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=100m tmpfs /mnt/ramdisk
```

### 2. Download and extract data files

Download the supplementary data files from the latest release:

**Ramdisk data files** (`rlgbs-ramdisk-data.tar.gz`):
Extract to `/mnt/ramdisk/`:
- `F_1920000.mat`: Pre-calculated IR view factor data
- `baseline_response_surface_v3_2.npy`: Baseline response surface (default parameter range)
- `baseline_response_surface_v3_wide_2.npy`: Baseline response surface (extended parameter range)

**Repository data files** (`rlgbs-repo-data.tar.gz`):
Extract to the project root directory:
- `eval/ppo-dryer-v3-122423-4msteps/`: Pre-trained RL model weights and observation normalizer
- `eval/cyd_baseline_points_256.npy`: Baseline evaluation points (256 2D grid)
- `eval/cyd_baseline_points_512.npy`: Baseline evaluation points (512 3D grid)
- `cydrums.cpython-311-x86_64-linux-gnu.so`: Pre-compiled drum dryer simulator binary

```bash
# Extract ramdisk data
cd /mnt/ramdisk
tar -xzf /path/to/rlgbs-ramdisk-data.tar.gz

# Extract repository data
cd /path/to/this/repo
tar -xzf /path/to/rlgbs-repo-data.tar.gz
```

### 3. Build Docker image

```bash
docker build -t rlgbs .
```

### 4. Run Docker container

```bash
docker run -it \
  --gpus all \
  -v $(pwd):/opt/RLGBS-source \
  -v /mnt/ramdisk:/mnt/ramdisk \
  --net=host \
  --ipc=host \
  rlgbs
```

## Reproducing Results

The following sections describe how to reproduce the experimental results reported in the paper.

### Configuration

Before running evaluations, configure the parameter range in `definitions.py`:

- Set `CYD_WIDE_DBMC_RANGE = False` for default initial condition grid ($\textrm{DBMC} \in [0.996, 1.196]$)
- Set `CYD_WIDE_DBMC_RANGE = True` for extended initial condition grid ($\textrm{DBMC} \in [0.9, 1.5]$)

### Evaluation Scripts

#### 1. Single evaluation runs

For one-off evaluation of specific initial conditions with different search strategies:

```bash
# Greedy search (1 beam)
python evaluate_single.py 1

# Beam search with 8 beams  
python evaluate_single.py 8

# Custom initial conditions [temp, speed, dbmc] normalized to [0,1]
python evaluate_single.py 32 --init 0.5 0.3 0.1
```

#### 2. Batch evaluation

For comprehensive evaluation over predefined initial condition grids:

```bash
# Run all methods (random, greedy, RL beam search) with default beam widths
python evaluate_batch.py --methods all

# Run only baseline methods
python evaluate_batch.py --methods random greedy

# Run RL beam search with custom beam widths
python evaluate_batch.py --methods rl_beam_search --beams 1 4 8 16 32

# Custom output directory
python evaluate_batch.py --methods all --output results/
```

**Available methods:**
- `random`: Random action baseline
- `greedy`: Greedy search baseline  
- `rl_beam_search`: RL model with beam search (1-256 beams)
- `all`: Run all methods

**Note**: Full batch evaluation may take several hours on a system with GPU and multi-core CPU.

## License

This code is released under the MIT License. However, the code under `hfgen/` is adapted from [HuggingFace Transformers](https://github.com/huggingface/transformers) and retains its original Apache 2.0 license.

The drum dryer simulation binary is provided as-is for reproducibility purposes.
