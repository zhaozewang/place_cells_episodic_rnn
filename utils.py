import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_response_map(response_map, random_cell=False, im_width=3, n_cols=5, n_rows=2):
    """ Visualize the response map """
    n_cells = response_map.shape[0]
    if random_cell:
        cell_indices = np.random.choice(n_cells, size=n_cols*n_rows, replace=False)
    else:
        cell_indices = range(n_cells)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(im_width*n_cols, im_width*n_rows))
    for i, ax in enumerate(axes.flat):
        if i < len(cell_indices):
            rm = response_map[cell_indices[i]]
            ax.imshow(rm, cmap='jet')
            min_fr, max_fr, mean_fr = np.nanmin(rm), np.nanmax(rm), np.nanmean(rm)
            ax.set_title(f'min: {min_fr:.2f}, max: {max_fr:.2f}\nmean: {mean_fr:.2f}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


class RatemapAggregator:
    def __init__(self, arena_map, device=None):
        """
        Class to accumulate partial data for rate-map computation.

        Parameters
        ----------
        arena_map : torch.Tensor or np.ndarray
            Map of shape (n_x, n_y), 0 for free space, 1 for walls, 
            used for figuring out dimensions and for masking.
        device : str or torch.device, optional
            The device on which to store the data (CPU or GPU). 
            If None, uses arena_map device if it's a torch.Tensor, 
            otherwise "cpu".
        """
        # If arena_map is numpy array, convert to torch
        if isinstance(arena_map, np.ndarray):
            arena_map = torch.as_tensor(arena_map)
        
        self.arena_map = arena_map
        self.dims = arena_map.shape  # (n_x, n_y)
        self.n_cells = None
        # Infer device
        if device is None:
            self.device = arena_map.device if arena_map.is_cuda else torch.device('cpu')
        else:
            self.device = torch.device(device)

    
    def init_counts(self):
        self.partial_sums = torch.zeros(
            (self.n_cells, *self.dims),
            dtype=torch.float32,
            device=self.device
        )
        # shape: (n_x, n_y)
        self.visit_counts = torch.zeros(
            self.dims,
            dtype=torch.float32,
            device=self.device
        )

    def update(self, states, coords):
        """
        Accumulate partial sums and visit counts from new data.

        Parameters
        ----------
        states : torch.Tensor or np.ndarray
            Shape=(n_batches, n_timesteps, n_cells) or (n_timesteps, n_batches, n_cells).
        coords : torch.Tensor or np.ndarray
            Shape=(n_batches, n_timesteps, 2) or (n_timesteps, n_batches, 2).
        """
        if self.n_cells is None:
            self.n_cells = states.shape[-1]
            self.init_counts()

        # Convert to torch if numpy
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states)
        if isinstance(coords, np.ndarray):
            coords = torch.as_tensor(coords)

        # Move to the same device
        states = states.float().to(self.device)
        coords = coords.float().to(self.device)

        # Ensure correct dtype
        states = states.float()
        coords = torch.round(coords).long()  # round and convert to long

        # Standardize shapes
        assert states.dim() == 3, "states must have 3 dims: (n_batches, n_timesteps, n_cells)"
        assert coords.dim() == 3, "coords must have 3 dims: (n_batches, n_timesteps, 2)"

        # Flatten
        coords = coords.reshape(-1, 2)     # (n_batches * n_timesteps, 2)
        states = states.reshape(-1, self.n_cells)  # (n_batches * n_timesteps, n_cells)

        # Flatten partial sums and visit_counts for fast index_add
        flat_sums = self.partial_sums.view(self.n_cells, -1)  # shape: (n_cells, n_x*n_y)
        flat_counts = self.visit_counts.view(-1)              # shape: (n_x*n_y)

        # Convert (row, col) coords into linear indices
        dims = self.dims
        flat_coords = coords[:, 0] * dims[1] + coords[:, 1]  # shape: (n_samples,)

        # Accumulate partial sums
        # states.T shape: (n_cells, n_samples)
        # so we add states.T to the flat_sums at the flattened coordinate indices
        flat_sums.index_add_(1, flat_coords, states.T)

        # Accumulate visit counts
        flat_counts.index_add_(
            0, 
            flat_coords, 
            torch.ones_like(flat_coords, dtype=torch.float32)
        )

    def get_ratemap(self):
        """
        Returns the final normalized firing fields (n_cells, n_x, n_y).
        Unvisited points (visit_count=0) will be NaN.
        """
        # Avoid division by zero by clamping
        denom = self.visit_counts.clamp(min=1.0)  # shape: (n_x, n_y)
        
        # Broadcasting: partial_sums shape (n_cells, n_x, n_y) / (n_x, n_y)
        ratemap = self.partial_sums / denom.unsqueeze(0)

        # Set unvisited areas to NaN
        mask_unvisited = (self.visit_counts == 0)
        ratemap[:, mask_unvisited] = float('nan')

        return ratemap

    def reset(self):
        """
        Reset the aggregator (clears all partial sums and counts).
        """
        self.partial_sums.zero_()
        self.visit_counts.zero_()
