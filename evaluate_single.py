#!/usr/bin/env python3
"""
Single evaluation script for RLGBS paper experiments.
Evaluates a specific initial condition using the trained RL model with configurable beam search.
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
from RLgen import PPOgen


def main(n_beams: int, init_conditions: Optional[list] = None):
    """Run single evaluation with specified beam width and initial conditions."""
    log_dir = "eval/ppo-dryer-v3-122423-4msteps"
    model_path = os.path.join(log_dir, "best_model")
    normalizer_path = os.path.join(log_dir, "vec_normalize.pkl")
    model = PPOgen.load(model_path)

    # Use provided initial conditions or default example
    if init_conditions is None:
        init_conditions = [0.4, 0.2, 0.2]  # [temp, speed, dbmc] normalized to [0, 1]
    
    prescribed_rstates = np.array(init_conditions)
    
    print(f"Running evaluation with {n_beams} beams")
    print(f"Initial conditions: temp={prescribed_rstates[0]:.3f}, speed={prescribed_rstates[1]:.3f}, dbmc={prescribed_rstates[2]:.3f}")

    t0 = time.time()
    candidates, selected_model_inputs = model.generate(
        num_beams=n_beams,
        num_return_sequences=min(8, max(1, n_beams//4)),
        remove_invalid_values=True,
        normalizer_path=normalizer_path,
        init_rstates=prescribed_rstates,
        gym_env_id="CydrumsEnv-v3"
    )
    t1 = time.time()
    
    print(f"Evaluation completed in {(t1 - t0):.3f}s")
    print(f"Best action sequence: {candidates}")
    print(f"Reward: {selected_model_inputs['all_rewards'].item():.4f} kJ m^-2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single evaluation run for RLGBS experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_single.py 1                     # Greedy search
  python evaluate_single.py 8                     # Beam search with 8 beams
  python evaluate_single.py 32 --init 0.4 0.2 0.2 # Custom initial conditions
        """
    )
    parser.add_argument("nbeams", type=int, help="Number of beams (1 for greedy search, 2-256 for beam search)")
    parser.add_argument("--init", nargs=3, type=float, metavar=("TEMP", "SPEED", "DBMC"),
                        help="Initial conditions [temp, speed, dbmc] normalized to [0,1]")
    
    args = parser.parse_args()
    
    if args.nbeams < 1 or args.nbeams > 256:
        parser.error("Number of beams must be between 1 and 256")
    
    init_cond = args.init if args.init else None
    if init_cond and any(x < 0 or x > 1 for x in init_cond):
        parser.error("Initial conditions must be normalized between 0 and 1")
    
    main(args.nbeams, init_cond)