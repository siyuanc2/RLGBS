import json
from multiprocessing import Pool, cpu_count
from typing import Any, List, Optional, Union, Tuple

import gymnasium as gym
import numpy as np
import redis
import torch
from definitions import CYD_WIDE_DBMC_RANGE
from gymnasium import spaces
# from hfgen.hf_logging import get_logger
from scipy.interpolate import interpn

from cydrums import cydrums, cyd_simresult # type: ignore

RESPONSE_SURFACE_FILE = "/mnt/ramdisk/baseline_response_surface_v3_wide_2.npy" if CYD_WIDE_DBMC_RANGE else "/mnt/ramdisk/baseline_response_surface_v3_2.npy"

def _redis_cached_env(conn: redis.Redis, input_ids_tuple: tuple, last_rstates: np.ndarray, n_section_size: int, n_actions: int, eval_mode: bool) -> Tuple[cyd_simresult, int]:
    # Repeated query to Redis using shorter and shorter keys
    for i in range(len(input_ids_tuple), 0, -1):
        if i == 1:
            # If only one action is left, create a new environment and simulate the action
            myenv = cydrums(steps_per_section=n_section_size, action_space=n_actions, eval_mode=eval_mode, use_wide_dbmc_range=CYD_WIDE_DBMC_RANGE)
            _ = myenv.reset(last_rstates[0], last_rstates[1], last_rstates[2])
            for ids in input_ids_tuple[:i]:
                if ids > -1:
                    ret = myenv.step(ids)
                    if ret.truncated:
                        raise ValueError("Simulation truncated at step 1!")
            break
        # Assemble the key, which is a tuple containing the last_rstates (rounded to 3 decimal places) followed by the input_ids_tuple
        key_tuple = tuple(np.round(last_rstates, 3)) + input_ids_tuple[:i]
        key = json.dumps(key_tuple)
        value = conn.get(key)
        if value is not None:
            myenv = cydrums(steps_per_section=n_section_size, action_space=n_actions, eval_mode=eval_mode, use_wide_dbmc_range=CYD_WIDE_DBMC_RANGE)
            serialized_env_state = np.frombuffer(value, dtype=np.float64)
            myenv.set_serialized_data(serialized_env_state.tolist())
            break
        # If value is None, continue to the next shorter key
    # Find out how many actions have been taken so far, which is the number of non-negative input_ids
    num_actions_taken = sum(1 for ids in input_ids_tuple[:i] if ids >= 0)
    # Check status of the environment. If already terminated or truncated, we do not need to simulate the remaining steps
    ret = myenv.status()
    # if ret.done or ret.truncated:
    #     return ret, num_actions_taken
    # simulate the remaining steps
    num_new_actions = 0
    for ids in input_ids_tuple[i:]:
        if ids > -1:
            num_new_actions += 1
            ret = myenv.step(ids)
            # set the new state to Redis
            key_tuple = tuple(np.round(last_rstates, 3)) + input_ids_tuple[:i+num_new_actions]
            key = json.dumps(key_tuple)
            value = np.array(myenv.get_serialized_data(), dtype=np.float64).tobytes()
            conn.set(key, value)
            # if ret.done or ret.truncated:
            #     break
    return ret, num_actions_taken + num_new_actions

def redis_inference_helper(args) -> List[dict]:
    i, num_processes, input_ids, last_rstates, n_section_size, n_actions, eval_mode = args
    batch_size = len(input_ids)
    # Get the range of the input_ids to be processed by this process
    start = i * (batch_size // num_processes)
    end = (i + 1) * (batch_size // num_processes) if i < num_processes - 1 else batch_size

    all_state_dicts = []
    conn = redis.Redis(host='localhost', port=6379, db=1 if CYD_WIDE_DBMC_RANGE else 0) # db=0 for normal range, db=1 for wide range
    for d in range(start, end):
        input_ids_tuple = input_ids[d]
        ret, num_actions_taken = _redis_cached_env(conn, input_ids_tuple, last_rstates, n_section_size, n_actions, eval_mode) # returns a cyd_simresult object
        # return the results as a dictionary for reward calculation later
        obs = np.array([ret.temp_top, ret.temp_bottom, ret.dbmc_top, ret.dbmc_bottom, ret.dbmc_avg, ret.speed], dtype=np.float32)
        state_dict = {"obs": obs, "info": {"cum_hf_all": ret.cumulative_heat_consumption, "dbmcavg": ret.dbmc_avg, "done": ret.done, "truncated": ret.truncated, "hf": ret.this_heat_consumption, "numsteps_sofar": num_actions_taken}}
        all_state_dicts.append(state_dict)
    conn.close()
    return all_state_dicts

class cyDrumEnv(gym.Env):
    def __init__(self, n_section_size: Optional[int] = 5, n_actions: Optional[int] = 11, 
                 eval_mode: Optional[bool] = False, fixed_num_steps: Optional[int] = 0,
                 penalize_extra_steps: Optional[bool] = True, use_baseline_reward: Optional[bool] = True, 
                 baseline_path: Optional[str] = RESPONSE_SURFACE_FILE, clip_reward: Optional[bool] = False):
        super().__init__()
        self.n_section_size = n_section_size
        self.n_actions = n_actions
        self.eval_mode = eval_mode
        self.baseline_path = baseline_path
        self._env = cydrums(steps_per_section=self.n_section_size, action_space=self.n_actions, eval_mode=self.eval_mode, use_wide_dbmc_range=CYD_WIDE_DBMC_RANGE)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=np.array([25.0, 25.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float32),
         high=np.array([100.0, 100.0, 1.5, 1.5, 1.5, 6.67], dtype=np.float32), dtype=np.float32)
        self.num_states = self._env.num_states
        self.last_rstates = None
        self.use_baseline_reward = use_baseline_reward
        self.clip_reward = clip_reward
        self.force_fixed_num_steps = fixed_num_steps > 0
        self.fixed_num_steps = fixed_num_steps
        self.penalize_extra_steps = penalize_extra_steps
        self.baseline_length_criterion = 999
        if self.use_baseline_reward:
            with open(baseline_path, "rb") as f:
                brs = np.load(f)
            self.baseline_points = (np.linspace(0, 1, brs.shape[0]), np.linspace(0, 1, brs.shape[1]), np.linspace(0, 1, brs.shape[2]))
            self.baseline_values = brs[:, :, :, 3]
            if self.penalize_extra_steps:
                self.baseline_num_steps = brs[:, :, :, 5] / n_section_size
        # self.logger = get_logger(__name__)
        self.cpu_count = cpu_count()
        self.numsteps_sofar = 0

    def _check_levels(self, input_level, tolerance=1e-5):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        """
        # We need the following line to seed self.np_random
        if seed is not None:
            self.seed(seed)
        # If initial conditions are specified, use as-is; otherwise draw random i.c. from rng and pass them to cyd environment
        if (options is not None) and ('rstates' in options.keys()):
            rstates = options['rstates'] # Expecting rstates is a 1x3 numpy vector
        else:
            # Warn if no prescribed random states are recognized
            print("No prescribed random states recognized, drawing from rng")
            rstates = self.np_random.random(size=3)
        # # Possible stopgap solution for segfault during extended runs
        # del self._env
        # self._env = cydrums(steps_per_section=self.n_section_size, action_space=self.n_actions, eval_mode=self.eval_mode, use_wide_dbmc_range=CYD_WIDE_DBMC_RANGE)
        ret = self._env.reset(rstates[0], rstates[1], rstates[2])
        self.last_rstates = rstates
        self._check_levels(self.last_rstates)
        self.baseline_length_criterion = 999
        if self.penalize_extra_steps:
            self.baseline_length_criterion = interpn(self.baseline_points, self.baseline_num_steps, self.last_rstates).item()
        obs = np.array([ret.temp_top, ret.temp_bottom, ret.dbmc_top, ret.dbmc_bottom, ret.dbmc_avg, ret.speed], dtype=np.float32)
        self.numsteps_sofar = 0
        info = {"cum_hf_all": ret.cumulative_heat_consumption, "dbmcavg": ret.dbmc_avg, "done": ret.done, "truncated": ret.truncated, "hf": ret.this_heat_consumption, "numsteps_sofar": 0}
        return obs, info

    def restart(self) -> np.ndarray:
        """
        Resets the environment to the previously used initial internal state, returning an initial observation and info.
        Useful in exploring multiple trajectories in parallel during beam search.
        """
        if self.last_rstates is not None:
            ret = self._env.reset(self.last_rstates[0], self.last_rstates[1], self.last_rstates[2])
        else:
            raise ValueError("Cannot restart environment: self.last_rstates is empty")
        obs = np.array([ret.temp_top, ret.temp_bottom, ret.dbmc_top, ret.dbmc_bottom, ret.dbmc_avg, ret.speed], dtype=np.float32)
        self.numsteps_sofar = 0
        return obs

    def redis_cached_batch_rollout(self, input_ids: torch.LongTensor, return_energy_as_reward: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run rollout for a batch of input_ids with parallelization. 
        Results are cached in Redis for future use.
        Output is always placed on CPU; manual conversion to CUDA is required.

        Parameters:
        - input_ids (np.ndarray): Batch of input IDs.
        - return_energy_as_reward (Optional[bool]): Whether to return energy as reward. Default is False.

        Returns:
        - tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing all observations, rewards, and dones.
        """
        batch_size, _ = input_ids.shape
        input_ids_array = input_ids.to("cpu").detach().numpy()
        cpp_input_ids = [tuple(int(idx) for idx in row) for row in input_ids_array] # json is such a PITA
        all_obs = torch.zeros(batch_size, self.num_states).to("cpu")
        all_rewards = torch.zeros(batch_size, 1).to("cpu")
        all_dones = torch.zeros(batch_size, 1, dtype=torch.bool).to("cpu")

        if batch_size > 8:
            num_processes = min(self.cpu_count, max(1, batch_size // 4)) # Ensure at least 4 sequences per process to avoid opening too many Redis connections (unless batch_size is smaller than 4)
            with Pool(processes=num_processes) as pool:
                results = pool.map(redis_inference_helper, [(i, num_processes, cpp_input_ids, self.last_rstates, self.n_section_size, self.n_actions, self.eval_mode) for i in range(num_processes)])
        else:
            # Dont use multiprocessing for small batches
            num_processes = 1
            results = [redis_inference_helper((i, num_processes, cpp_input_ids, self.last_rstates, self.n_section_size, self.n_actions, self.eval_mode)) for i in range(num_processes)]

        # Combine the results from all processes
        for i in range(num_processes):
            start = i * (batch_size // num_processes)
            end = (i + 1) * (batch_size // num_processes) if i < num_processes - 1 else batch_size
            for j in range(end-start):
                reward, done, truncated = self.get_reward(results[i][j]["info"])
                all_rewards[start+j] = reward
                all_dones[start+j] = done or truncated
                all_obs[start+j, :] = torch.from_numpy(results[i][j]["obs"])

        return all_obs, all_rewards, all_dones

    def get_reward(self, info_dict: dict) -> Tuple[float, bool, bool]:
        """
        Calculate the reward based on the info_dict, also return done and truncated flags.
        """
        numsteps_sofar = info_dict["numsteps_sofar"]
        done = info_dict["done"]
        truncated = info_dict["truncated"]
        reward = 0
        if self.force_fixed_num_steps:
            # Force the simulation to continue until fixed_num_steps is reached
            if numsteps_sofar < self.fixed_num_steps:
                done = False
        if done:
            if self.use_baseline_reward:
                # Reward function used for v2 and v3 simulation
                baseline_criterion = interpn(self.baseline_points, self.baseline_values, self.last_rstates)
                reward = (baseline_criterion - info_dict["cum_hf_all"]).item()
            else:
                # Obsolete reward function (v1) based on abs(DBMC_final - DBMC_target)
                if info_dict["dbmcavg"] >= 0.18:
                    reward = 100. * info_dict["dbmcavg"] - 18. # end DBMC = 0.2 -> reward = 2, 0.19 -> 1, <0.18 -> 0
                else:
                    reward = 0
        if truncated or np.isnan(info_dict["dbmcavg"]):
            # Handle simulation instability due to extreme operating conditions
            reward -= 100.
            truncated = True
        if self.penalize_extra_steps:
            # Penalize extra steps taken by the agent to reach target DBMC
            reward -= np.fmax(numsteps_sofar - np.rint(self.baseline_length_criterion), 0) * info_dict["hf"] * 0.5
        if self.clip_reward:
            reward = np.clip(reward, -20., 20.)
        if np.isnan(reward):
            reward = -100.
        return reward, done, truncated

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics using the agent actions.
        """
        ret = self._env.step(action)
        obs = np.array([ret.temp_top, ret.temp_bottom, ret.dbmc_top, ret.dbmc_bottom, ret.dbmc_avg, ret.speed], dtype=np.float32)
        self.numsteps_sofar += 1
        info = {"cum_hf_all": ret.cumulative_heat_consumption, "dbmcavg": ret.dbmc_avg, "done": ret.done, "truncated": ret.truncated, "hf": ret.this_heat_consumption, "numsteps_sofar": self.numsteps_sofar}
        reward, done, truncated = self.get_reward(info)
        return obs, reward, done, truncated, info

    def seed(self, seed: int) -> None:
        """
        Seed the rng gym.Env.np_random using the supplied seed.
        """
        super().reset(seed=seed)

    def close(self) -> None:
        """
        Close the environment.
        """
        del self._env

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = cyDrumEnv()
    check_env(env, warn=True)
