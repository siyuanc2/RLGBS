#!/usr/bin/env python3
"""
Batch evaluation script for RLGBS paper experiments.
Evaluates all methods (random, greedy, beam search) over predefined initial condition grids.
"""

import argparse
import os
import time
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from RLgen import PPOgen
from definitions import CYD_WIDE_DBMC_RANGE, BS_MAX_REFINE_STEPS


def run_random_baseline(shuffled_rstates: np.ndarray) -> List[Dict[str, Any]]:
    """Run random action baseline evaluation using direct gymnasium environment."""
    print("Running random action baseline...")
    
    response_surface_path = "/mnt/ramdisk/baseline_response_surface_v3_wide_2.npy" if CYD_WIDE_DBMC_RANGE else "/mnt/ramdisk/baseline_response_surface_v3_2.npy"
    with open(response_surface_path, "rb") as f:
        brs = np.load(f)
    baseline_values = brs[:, :, :, 3]
    
    # Use gymnasium environment directly for random actions
    env = gym.make("CydrumsEnv-v3")
    np.random.seed(42)  # For reproducible random actions
    
    results = []
    for init_temp, init_speed, init_dbmc in shuffled_rstates:
        # Determine grid indices
        i = int(init_temp * 30)
        j = int(init_speed * 30) 
        k = int(init_dbmc * 30)
        
        t0 = time.time()
        env.reset(options={"rstates": np.array([init_temp, init_speed, init_dbmc])})
        
        done = False
        truncated = False
        numsteps_sofar = 0
        action_sequence = []
        
        while not done and not truncated:
            # Choose random action from valid action space (0-10 for temperature control)
            action = np.random.randint(0, 11)
            obs, reward, done, truncated, info = env.step(action)
            action_sequence.append(action)
            numsteps_sofar += 1
        
        t1 = time.time()
        
        result = {
            "method": "random",
            "init_temp": round(init_temp, 4),
            "init_speed": round(init_speed, 4), 
            "init_dbmc": round(init_dbmc, 4),
            "time": round(t1 - t0, 3),
            "reward": round(reward, 4),
            "energy": round(baseline_values[i, j, k] - reward, 4),
            "baseline_value": round(baseline_values[i, j, k], 4),
            "candidates": action_sequence,
            "numsteps": numsteps_sofar
        }
        results.append(result)
        
        print(f"Random - Init: {init_temp:.3f}, {init_speed:.3f}, {init_dbmc:.3f} -> Reward: {reward:.4f}, Steps: {numsteps_sofar}")
    
    return results


def run_greedy_baseline(shuffled_rstates: np.ndarray) -> List[Dict[str, Any]]:
    """Run greedy search baseline evaluation."""
    print("Running greedy search baseline...")
    
    env = gym.make("CydrumsEnv-v3")
    results = []
    
    for init_temp, init_speed, init_dbmc in shuffled_rstates:
        t0 = time.time()
        env.reset(options={"rstates": np.array([init_temp, init_speed, init_dbmc])})
        
        done = False
        truncated = False
        numsteps_sofar = 0
        
        while not done and not truncated:
            obs, reward, done, truncated, info = env.step(10)  # Always choose max temperature
            numsteps_sofar += 1
        
        t1 = time.time()
        
        result = {
            "method": "greedy",
            "init_temp": round(init_temp, 4),
            "init_speed": round(init_speed, 4),
            "init_dbmc": round(init_dbmc, 4),
            "time": round(t1 - t0, 3),
            "reward": round(reward, 4),
            "numsteps": numsteps_sofar
        }
        results.append(result)
        
        print(f"Greedy - Init: {init_temp:.3f}, {init_speed:.3f}, {init_dbmc:.3f} -> Reward: {reward:.4f}, Steps: {numsteps_sofar}")
    
    return results


def run_rl_beam_search(shuffled_rstates: np.ndarray, beam_widths: List[int]) -> List[Dict[str, Any]]:
    """Run RL model with beam search evaluation."""
    print(f"Running RL model with beam search widths: {beam_widths}")
    
    response_surface_path = "/mnt/ramdisk/baseline_response_surface_v3_2.npy"
    with open(response_surface_path, "rb") as f:
        brs = np.load(f)
    baseline_points = (np.linspace(0, 1, brs.shape[0]), np.linspace(0, 1, brs.shape[1]), np.linspace(0, 1, brs.shape[2]))
    baseline_values = brs[:, :, :, 3]
    
    log_dir = os.path.join("eval/ppo-dryer-v3-122423-4msteps")
    model_path = os.path.join(log_dir, "best_model.zip")
    normalizer_path = os.path.join(log_dir, "vec_normalize.pkl")
    model = PPOgen.load(model_path)

    results = []
    cumulative_time = 0.0
    
    # Evaluate subset of initial conditions for full beam search
    for i, init_temp in enumerate(baseline_points[0]):
        if abs(init_temp - 0.4) > 1e-5:  # Focus on specific temperature
            continue
        for j, init_speed in enumerate(baseline_points[1]):
            if j % 2 != 0:  # Skip every other speed point
                continue
            for k, init_dbmc in enumerate(baseline_points[2]):
                if k % 4 != 0:  # Skip every other DBMC point
                    continue
                
                for n_beams in beam_widths:
                    t0 = time.time()
                    candidates, selected_model_inputs = model.generate(
                        num_beams=n_beams,
                        num_return_sequences=min(8, max(1, n_beams//4)),
                        remove_invalid_values=True,
                        normalizer_path=normalizer_path,
                        init_rstates=np.array([init_temp, init_speed, init_dbmc]),
                        gym_env_id="CydrumsEnv-v3"
                    )
                    t1 = time.time()
                    cumulative_time += t1 - t0
                    
                    reward = selected_model_inputs["all_rewards"].squeeze().item()
                    
                    result = {
                        "method": "rl_beam_search",
                        "init_temp": round(init_temp, 4),
                        "init_speed": round(init_speed, 4),
                        "init_dbmc": round(init_dbmc, 4),
                        "n_beams": n_beams,
                        "time": round(t1 - t0, 2),
                        "cumulative_time": round(cumulative_time, 2),
                        "candidates": [x for x in candidates.squeeze().cpu().numpy().tolist() if x >= 0 and x <= 10],
                        "reward": round(reward, 4),
                        "energy": round(baseline_values[i, j, k] - reward, 4),
                        "baseline_value": round(baseline_values[i, j, k], 4)
                    }
                    results.append(result)
                    
                    print(f"RL-BS({n_beams}) - Init: {init_temp:.3f}, {init_speed:.3f}, {init_dbmc:.3f} -> Reward: {reward:.4f}")
    
    return results


def main(methods: List[str], beam_widths: List[int], output_dir: str):
    """Run batch evaluation for specified methods."""
    print(f"Batch evaluation using CYD_WIDE_DBMC_RANGE = {CYD_WIDE_DBMC_RANGE}")
    print(f"Methods to evaluate: {methods}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation points
    points_file = "eval/cyd_baseline_points_256.npy" if "random" in methods else "eval/cyd_baseline_points_512.npy"
    with open(points_file, "rb") as f:
        shuffled_rstates = np.load(f)
    
    all_results = []
    
    if "random" in methods:
        all_results.extend(run_random_baseline(shuffled_rstates))
    
    if "greedy" in methods:  
        all_results.extend(run_greedy_baseline(shuffled_rstates))
    
    if "rl_beam_search" in methods:
        all_results.extend(run_rl_beam_search(shuffled_rstates, beam_widths))
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(output_dir, f"batch_evaluation_results.jsonl")
        df.to_json(output_file, orient="records", lines=True)
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        if "rl_beam_search" in methods:
            rl_results = df[df["method"] == "rl_beam_search"]
            print(f"\nRL Beam Search Summary:")
            print(f"Best reward: {rl_results['reward'].max():.4f}")
            print(f"Mean reward: {rl_results['reward'].mean():.4f}")
            print(f"Total evaluation time: {rl_results['cumulative_time'].max():.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation for RLGBS experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_batch.py --methods random greedy        # Run baselines only
  python evaluate_batch.py --methods rl_beam_search       # Run RL with default beam widths
  python evaluate_batch.py --methods all --beams 1 4 8 16 # Run all methods with custom beam widths
        """
    )
    parser.add_argument("--methods", nargs="+", 
                        choices=["random", "greedy", "rl_beam_search", "all"],
                        default=["all"],
                        help="Methods to evaluate")
    parser.add_argument("--beams", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help="Beam widths for RL beam search")
    parser.add_argument("--output", default="output",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Expand "all" to individual methods
    if "all" in args.methods:
        methods = ["random", "greedy", "rl_beam_search"]
    else:
        methods = args.methods
    
    # Validate beam widths
    if any(b < 1 or b > 256 for b in args.beams):
        parser.error("Beam widths must be between 1 and 256")
    
    main(methods, args.beams, args.output)