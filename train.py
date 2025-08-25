# pylint: disable=import-error
"""
Main training script for the PPO agent.
"""

import os
import time
from typing import Callable

import gymnasium as gym
import numpy as np
# from cygymwrapper import cyDrumEnv
# from smart_dryer_wrapper import SmartDryerEnv
from definitions import *
# import pybullet_envs
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecNormalize)


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        # env = gym.make("SmartDryerEnv-v1") # Dont use this, ensure that the environment is created with training mode=True
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        """
        Those variables will be accessible in the callback (they are defined in the base class)
        The RL model
            self.model = None  # type: BaseAlgorithm
        An alias for self.model.get_env(), the environment used for training
            self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        Number of time the callback was called
            self.n_calls = 0  # type: int
            self.num_timesteps = 0  # type: int
        local and global variables
            self.locals = None  # type: Dict[str, Any]
            self.globals = None  # type: Dict[str, Any]
        The logger object, used to report things in the terminal
            self.logger = None  # stable_baselines3.common.logger
        Sometimes, for event callback, it is useful to have access to the parent object
            self.parent = None  # type: Optional[BaseCallback]
        """
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                # mean_reward = np.mean(y[-100:])
                mean_reward = np.mean(self.training_env.unnormalize_reward(y[-100:]))
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    self.logger.record("Raw eps R", mean_reward)
                    # self.logger.dump(self.num_timesteps)

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    self.training_env.save(self.stats_path)
        return True

def main():
    """
    Train the agent on the dryer environment.
    """
    num_cpu = 30
    log_dir = os.path.join(MODEL_DIR, "ppo-dryer-v1-20240301")
    os.makedirs(log_dir, exist_ok=True)
    random_seed_base = np.random.randint(450, 12450)
    vecenv = SubprocVecEnv([make_env(i+random_seed_base) for i in range(num_cpu)])

    if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
        print("Using existing vec_normalize.pkl")
        normvecenv = VecMonitor(VecNormalize.load(load_path=os.path.join(log_dir, "vec_normalize.pkl"), venv=vecenv), log_dir)
    else:
        normvecenv = VecMonitor(VecNormalize(vecenv, norm_obs=True, norm_reward=True), log_dir)

    if os.path.exists(os.path.join(log_dir, "best_model.zip")):
        print("Using existing best_model.zip")
        model = PPO.load(os.path.join(log_dir, "best_model.zip"), env=normvecenv, learning_rate=1e-4)
    else:
        model = PPO('MlpPolicy', normvecenv, verbose=1, target_kl=0.01, learning_rate=3e-4, tensorboard_log=os.path.join(log_dir, "ppo-dryer"))

    callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)

    n_timesteps = 1e7 # 1e7 for v1
    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps, callback=callback, tb_log_name="PPO_v1", reset_num_timesteps=False)
    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

    # Save the VecNormalize statistics when saving the agent
    # model.save(log_dir + "ppo_cygym_final")
    model.save(os.path.join(log_dir, "ppo_model_final.zip"))
    normvecenv.save(os.path.join(log_dir, "vec_normalize_final.pkl"))

if __name__ == "__main__":
    main()
