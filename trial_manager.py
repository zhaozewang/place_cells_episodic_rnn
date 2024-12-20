import torch
import numpy as np
from functools import wraps
from trial import Trial, Trajectory, Arena

# ENVIRONMENT CONFIGURATION
ENV = "local"


# DECORATORS
# ******************************************************************************
def check_init(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._trial is None:
            raise ValueError("Trial not initialized. Please run new_trial() method first.")
        # If the trial is initialized, run the method
        return method(self, *args, **kwargs)
        if self._arena is None:
            raise ValueError("Arena not initialized. Please run initialize_gym() method first.")
    return wrapper


class TrialManager():
    """
    The TrialManager class is responsible for managing the trials in the experiment.
    It generates the trials and converts the input data into episodic bouts.
    """
    def __init__(self, gym_cfg: dict, expt_cfg: dict, device: torch.device):
        """
        Initialize the TrialManager.

        Parameters:
            gym_cfg (dict): The configuration dictionary for the gym.
            expt_cfg (dict): The configuration dictionary for the experiment.
        """
        # Store the configuration parameters
        self.gym_cfg = gym_cfg
        self.expt_cfg = expt_cfg
        self.device = device

        # Initialize the variables
        self._cur_s = 0
        self._trial = None
        self._arena = None
        self._sensory_map = None

        # Initialize the gym
        self.initialize_gym()

    @property
    def trial(self):
        return self._trial

    @property
    def arena(self):
        return self._arena

    def initialize_gym(self):
        if ENV == "local":
            # If running locally, use the full rtgym package
            from rtgym import RatatouGym
            self._gym = RatatouGym(
                temporal_resolution=self.gym_cfg["temporal_resolution"],
                spatial_resolution=self.gym_cfg["spatial_resolution"],
            )
            self._gym.init_arena_map(shape="rectangle")
            self._gym.set_sensory_from_profile(self.gym_cfg["sensory"])
            self._gym.set_behavior_from_profile({
                "velocity_mean": 5,
                "velocity_sd": 0.2,
                "random_drift_magnitude": 0.05,
                "switch_direction_prob": 0.3,
                "switch_velocity_prob": 0.2
            })
            self._arena = Arena(arena_map=self._gym.arena_map)
            self._sensory_map = self._gym.agent.sensory.sensories["default_sensory_group"].sm_responses
        else:
            # If running as public version, use the lite version of rtgym
            from rtgym_lite import RataGymLite
            # TODO: Implement the initialization of the lite version of the gym

    def new_trial(self):
        if ENV == "local":
            self._gym.trial.new_trial(
                duration=self.expt_cfg["trial_duration"], 
                batch_size=1  # Gen a single trajectory, batched by pieces of episodic memory
            )
            trajectory = Trajectory(
                coords_float=self._gym.trial.coords.f,
                displacements=self._gym.trial.disps
            )
        else:
            # TODO: Implement the new_trial method for the lite version of the gym
            pass
        self._trial = Trial(
            trajectory=trajectory, 
            sensory_map=self._sensory_map,
            expt_cfg=self.expt_cfg,
            temporal_resolution=self.gym_cfg["temporal_resolution"],
            device=self.device
        )
