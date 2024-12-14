import rtgym

class TrialManager():
    def __init__(self):
        pass

    def generate_trial(self, cfg):
        gym = rtgym.RatatouGym(
            temporal_resolution=cfg["temporal_resolution"],
            spatial_resolution=cfg["spatial_resolution"],
        )
        gym.set_sensory_from_profile(cfg["sensory"])
        gym.set_behavior_from_profile(cfg["behavior"])
        return gym