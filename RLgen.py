from hfgen.utils import GenerationMixin
from stable_baselines3 import PPO


class PPOgen(GenerationMixin, PPO):
    """
    Wrapper class for mixing sb3 PPO class with an external generation module
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
