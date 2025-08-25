import pickle
from copy import deepcopy
from typing import Union

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd


class AutoNormalizer:
    """
    sb3 autonormalizer, used to auto-normalize env obs and rewards as needed.
    """
    def __init__(
        self,
        load_path: str,
    ):
        """
        Initialize the AutoNormalizer object.

        Args:
            load_path (str): The path to the saved normalization statistics.
        """
        with open(load_path, 'rb') as f:
            sb3_vec_normalize = pickle.load(f)
        self.norm_obs = sb3_vec_normalize.norm_obs
        self.norm_reward = sb3_vec_normalize.norm_reward
        self.obs_rms = deepcopy(sb3_vec_normalize.obs_rms) if self.norm_obs else None
        self.ret_rms = deepcopy(sb3_vec_normalize.ret_rms) if self.norm_reward else None
        self.epsilon = sb3_vec_normalize.epsilon
        self.gamma = sb3_vec_normalize.gamma
        self.clip_obs = sb3_vec_normalize.clip_obs
        self.clip_reward = sb3_vec_normalize.clip_reward
        return

    def normalize_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Normalize observations using this AutoNormalizer's observations statistics.
        Calling this method does not update statistics.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): The observations to be normalized.

        Returns:
            np.ndarray: The normalized observations.
        """
        if isinstance(obs, torch.Tensor):
            return torch.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        obs_ = deepcopy(obs)
        obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
        return obs_

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper method to normalize observation.

        Args:
            obs (np.ndarray): The observation to be normalized.
            obs_rms (RunningMeanStd): The associated statistics.

        Returns:
            np.ndarray: The normalized observation.
        """
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this AutoNormalizer's rewards statistics.
        Calling this method does not update statistics.

        Args:
            reward (np.ndarray): The rewards to be normalized.

        Returns:
            np.ndarray: The normalized rewards.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        TODO: to be implemented
        """
        raise NotImplementedError

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        TODO: to be implemented
        """
        raise NotImplementedError
