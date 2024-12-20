import numpy as np
import torch


# DATA CLASSES
# ******************************************************************************
class Coord:
    """
    The Coord class is a wrapper for the coordinates of the agent. Could be multiple
    coordinates or a single coordinate.
    """
    def __init__(self, coord_float: np.ndarray):
        self.coord_float = coord_float
        self.coord = coord_float.astype(int)
    
    def __repr__(self):
        return repr(self.coord)
    
    @property
    def f(self):
        """
        Returns the coordinates as float values.
        """
        return self.coord_float
    
    def __getitem__(self, key):
        return self.coord[key]

    def __getattr__(self, attr):
        # Delegate any unknown attribute to the coords array
        return getattr(self.coord, attr)


# TRIAL DATA CLASS
# ******************************************************************************
class Trajectory:
    """
    Data class to hold trial data. It stores the coordinates and displacements of the agent during the trial.
    """
    def __init__(self, coords_float: np.ndarray, displacements: np.ndarray):
        """
        Initialize the Trajectory object with the given data.

        Parameters:
            - coords_float: Coordinates of the agent as float values.
            - displacements: Displacements of the agent between coordinates.
                             The displacement at the last step is set to 0.
        """
        # Check if the input data are all numpy arrays
        assert isinstance(coords_float, np.ndarray), "coords_float must be a numpy array"
        assert isinstance(displacements, np.ndarray), "displacements must be a numpy array"

        # Check if the input data all have three dimensions, (n_batch, n_time, n_features)
        assert coords_float.ndim == 3, "coords_float must have three dimensions (n_batch, n_time, n_features)"
        assert displacements.ndim == 3, "displacements must have three dimensions (n_batch, n_time, n_features)"

        # Check if the time dimension of the input data are the same
        assert coords_float.shape[1] == displacements.shape[1], "coords_float and displacements must have the same time dimension"

        # Store the input data
        self.coords = Coord(coords_float)
        self.disps = displacements

    def copy(self):
        """ Returns a deep copy of the Trajectory object. """
        return Trajectory(
            coords_float=self.coords.f.copy(),
            displacements=self.disps.copy()
        )


class Trial:
    """
    Handles the iteration and segmentation of trajectory data.
    This class takes a Trajectory object and handles iteration over the trajectory.
    """
    def __init__(self, 
                 trajectory: Trajectory,
                 sensory_map: np.ndarray,
                 temporal_resolution: int, 
                 expt_cfg: dict,
                 device: str = "cpu"
        ):
        """
        Parameters:
            - trajectory (Trajectory): A Trajectory object containing the coordinates and displacements.
            - sensory_map (np.ndarray): The sensory map of the environment.
            - temporal_resolution (int): Temporal resolution in milliseconds (ms).
            - device (str): The device to use for the data (default: "cpu").
        """
        # Store the parameters
        self.trajectory = trajectory
        self.sensory_map = sensory_map
        self.expt_cfg = expt_cfg
        self.device = device
        self.temporal_resolution = temporal_resolution

        # Generate the sensory response for the trajectory
        self.sensory_responses = self.generate_sensory_response()

        # For iteration
        self.total_ts = trajectory.coords.shape[1]  # Total number of timesteps
        self.current_time = 0  # Time (in seconds) to start iteration
        self.current_ts = 0  # Corresponding timestep index

        self.window_size_ts = self._to_ts(self.expt_cfg["window_size"])
        self.window_stride_ts = self._to_ts(self.expt_cfg["window_stride"])
        self._n_updates = (self.total_ts-self.window_size_ts) // self.window_stride_ts

    @property
    def n_updates(self):
        """ Number of updates to perform """
        return self._n_updates

    def generate_sensory_response(self):
        """ Generate the sensory response for the trajectory """
        traj = self.trajectory
        sensory_responses = self.sensory_map[:, traj.coords[..., 0], traj.coords[..., 1]]
        sensory_responses = sensory_responses.transpose(1, 2, 0)
        sensory_responses = torch.tensor(sensory_responses, dtype=torch.float32).to(self.device)
        return sensory_responses
    
    def _to_ts(self, s: float) -> int:
        """Convert the time in seconds to the corresponding timestep (integer index)."""
        return int(s * 1e3 / self.temporal_resolution)
    
    def _to_s(self, ts: int) -> float:
        """Convert the timestep to the corresponding time in seconds."""
        return ts * self.temporal_resolution / 1e3

    def __iter__(self):
        """Reset the iterator to the start of the trial and return itself."""
        self.current_time = 0
        self.current_ts = 0
        return self

    def __next__(self):
        """Return the next windowed segment of trajectory data."""
        if self.current_ts >= self.total_ts:
            raise StopIteration

        # Calculate start and end timesteps for the current window
        start_ts = self.current_ts
        end_ts = start_ts + self.window_size_ts

        # Check if we have enough data for the window
        if end_ts > self.total_ts:
            raise StopIteration

        # Get episodic memory bouts
        memory_bouts = self.get_episodic_memory(start_ts, end_ts)

        # Move to the next position for the next window
        self.current_ts += self.window_stride_ts

        return memory_bouts

    def get_episodic_memory(self, start_ts: int, end_ts: int):
        """
        Get the episodic memory at the specified time step.
        """
        alpha = self.expt_cfg["sampling_alpha"]
        beta = self.expt_cfg["sampling_beta"]
        epi_dur_ts = self._to_ts(self.expt_cfg["episode_duration"])
        n_epi = self.expt_cfg["batch_size"]

        # Calculate the range of available timesteps to sample from
        available_ts = np.arange(start_ts, end_ts - epi_dur_ts)
        
        # Calculate the sampling probabilities according to the power-law distribution
        choose_prob = np.linspace(0, 1, len(available_ts)) ** alpha + beta
        choose_prob /= choose_prob.sum()
        
        # Sample starting indices for episodic bouts
        idx = np.random.choice(available_ts, size=n_epi, p=choose_prob).astype(int)
        
        # Extract the episodic memory bouts from the trajectory
        memory_bouts = torch.cat(
            [self.sensory_responses[:, i:i + epi_dur_ts] for i in idx], dim=0
        )
        
        return memory_bouts


class Arena:
    def __init__(self, arena_map):
        """
        Arena class
        
        Parameters:
            - arena_map (np.ndarray): arena map (0: free space, 1: wall)
        """
        assert isinstance(arena_map, np.ndarray), "arena map must be a numpy array"
        self._arena_map = arena_map
        self.dimensions = self._arena_map.shape
        self.free_space = np.argwhere(self._arena_map == 0)

    @property
    def arena_height(self):
        return self.dimensions[0]

    @property
    def arena_width(self):
        return self.dimensions[1]

    @property
    def map(self):
        return self._arena_map

    @property
    def arena_map(self):
        return self._arena_map

    @property
    def inv_arena_map(self):
        return 1 - self._arena_map

    def generate_random_pos(self, size):
        """ Get random positions in the arena """
        return self.free_space[np.random.choice(self.free_space.shape[0], size=size, replace=False)]

    def validate_index(self, pos):
        """ Check if the position is in the arena """
        if len(pos.shape) == 1:
            pos = pos[np.newaxis, :]
        # check dimension
        assert pos.shape[1] == 2, "pos must be a 2D array"
        # check if the indices are defined
        is_negative = np.all(pos >= 0, axis=1)
        is_exceed = np.all(pos < self.dimensions, axis=1)
        valid_idx = np.logical_and(is_negative, is_exceed)
        is_wall = np.full(pos.shape[0], True)
        is_wall[valid_idx] = self._arena_map[tuple(pos[valid_idx].T)] == 1
        return np.logical_not(is_wall)
