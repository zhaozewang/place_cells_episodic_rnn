import numpy as np
from numpy.random import default_rng
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

class WeakSMCell:
    """ Spatially modulated non-grid/place responses """
    sens_type = 'weak_sm_cell'
    def __init__(self, arena_map, n_cells, seed=None, **kwargs):
        """
        @kwarg arena_map: arena map, shape (arena_width, arena_height)
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg seed: (optional) seed for random number generator, for reproducibility
        @kwarg sigma: sigma of the gaussian filter(s) (cm), must be an int
        @kwarg ssigma: how the sigma for each cell varies, if 0, all cells have the same sigma (spatial tuning)
        @kwarg magnitude: mean magnitude of the cell responses

        @attr response_map: spatially modulated responses, shape (n_cells, *arena_dimensions)
        """
        # Base class parameters
        self.arena_map = arena_map
        self.n_cells = n_cells

        # Initialize random number generator. If None, this will be the current
        # time, so it will be `random` every time
        self.rng = default_rng(seed)

        # Additional parameters for WeakSMCell
        self.sigma = kwargs.get('sigma', 8)
        self.ssigma = kwargs.get('ssigma', 0)
        self.magnitude = kwargs.get('magnitude', None)
        
        # Check parameters and initialize responses
        self._check_params()
        self._init_response_map()

    def _check_params(self):
        """ Check if parameters are valid """
        assert self.n_cells > 0, "n_cells <= 0"
        assert self.sigma > 0, "sigma <= 0"
        assert self.ssigma >= 0, "ssigma < 0"
        assert type(self.sigma) == int, "sigma must be an int"
        assert type(self.ssigma) == int, "ssigma must be an int"
        # sigma can be a list or an integer
        if isinstance(self.sigma, int):
            assert self.sigma > 0, "sigma <= 0"
        elif isinstance(self.sigma, list):
            for s in self.sigma:
                assert s > 0, "sigma <= 0"

    def _init_response_map(self):
        """ Initialize response_map """
        # border padding is also included in the response field
        self.response_map = np.zeros((self.n_cells, *self.arena_map.shape))
        self.response_map = self._generate_smcells()

    def _generate_smcells(self):
        """ 
        Generate a spatially modulated non-grid/place cell response field 
        """
        cells = self.rng.normal(0, 1, (self.n_cells, *self.arena_map.shape))
        # filter each cell response field with a 2d gaussian filter
        if self.ssigma > 0:
            mean, std = self.sigma, self.ssigma
            a = (0 - mean) / std  # 0 is the lower bound of the truncated normal distribution
            b = (100 - mean) / std  # 100 is the upper bound of the truncated normal distribution
            sigma_distribution = truncnorm(a=a, b=b, loc=mean, scale=std)
            sigma_list = sigma_distribution.rvs(self.n_cells, random_state=self.rng)
        else:
            sigma_list = np.ones(self.n_cells) * self.sigma
        for i in range(self.n_cells):
            cell = gaussian_filter(cells[i], sigma_list[i], mode='constant')
            cell = (cell - cell.min()) / (cell.max() - cell.min())  # normalize to [0, 1]
            cells[i] = cell
        
        # Scale cells to have desired mean magnitude instead of max magnitude
        if self.magnitude is not None:
            # Calculate current mean for each cell
            cell_means = cells.mean(axis=(1, 2))
            # Create scaling factors to achieve target mean magnitude
            scaling_factors = self.magnitude / cell_means
            # Apply scaling factors to each cell (broadcasting across spatial dimensions)
            cells = cells * scaling_factors[:, np.newaxis, np.newaxis]
        
        return cells # (n, *arena_dims)

    def get_response(self, coord):
        """
        Get response_map

        Parameters:
            - coord: (n_batch, n_timesteps, 2) tensor of coordinates

        Returns:
            - response_map: spatially modulated responses. The response_map are of shape (n_cells, *arena_dimensions).
                            After indexing, it will be of shape (n_cells, n_batch, n_timesteps).
                            When return, reshape it to (n_batch, n_timesteps, n_cells).
        """
        # convert (possibly float) coordinates to integer indices
        coord = np.round(coord).astype(int)
        res = self.response_map[:, coord[..., 0], coord[..., 1]]  # (n_cells, n_batch, n_timesteps)
        return res.transpose(1, 2, 0)  # (n_batch, n_timesteps, n_cells)
