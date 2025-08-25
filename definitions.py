"""
Global config file for training and evaluation.
"""
import os

from gymnasium.envs.registration import register

# Environments
register(
	id="CydrumsEnv-v3",
	entry_point="drum_dryer_wrapper:cyDrumEnv",
	max_episode_steps=12,
)

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(ROOT_DIR, "eval")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
CYD_WIDE_DBMC_RANGE = False

# Hyperparameters
ENV_SECTION_SIZE = 5
ENV_NUM_ACTIONS = 11
ENV_NUM_ACTION_COMBINATIONS = 11
ENV_NUM_RSTATES = 3 # Number of random variables in the environment state
# Decoding
BS_LENGTH_PENALTY = 1.0
BS_MAX_KEEP_HISTORY_LENGTH = 14 # max length 12 + 1 init padding token + 1 end token
BS_MAX_REFINE_STEPS = 0 # 0 - no refinement, 1 - refine last one step, 2 - refine last two steps

