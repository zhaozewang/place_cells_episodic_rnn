import torch
import torch.nn as nn
import nn4n
import nn4n.layer as nnl

class PlaceCellsEpisodicRNN(nnl.RNNLayer):
    """
    The Place Cells Episodic RNN is a recurrent neural network implementation.
    This class serves as a constructor for a custom RNN using nn4n.
    """
    def __init__(self, cfg: dict):
        """
        Initialize the Place Cells Episodic RNN.
        
        Parameters:
            cfg (dict): A dictionary containing the configuration parameters for the RNN.
                The dictionary should contain the following keys:
                    - input_dim (int): The dimensionality of the input data.
                    - hidden_dim (int): The dimensionality of the hidden layer.
                    - output_dim (int): The dimensionality of the output data.
                    - alpha (float): The learning rate of the hidden layer. Between 0 and 1.
                    - preact_noise (float): The standard deviation of the noise added to the pre-activation of the hidden layer.
                    - postact_noise (float): The standard deviation of the noise added to the post-activation of the hidden layer
        """
        # Create the hidden layer
        hidden_layer = nnl.HiddenLayer(
            input_layer=nnl.LinearLayer(input_dim=cfg.input_dim, output_dim=cfg.hidden_dim),
            linear_layer=nnl.LinearLayer(input_dim=cfg.hidden_dim, output_dim=cfg.hidden_dim),
            activation=nn.ReLU(),
            alpha=cfg.alpha,
            learn_alpha=False,  # Learning alpha is disabled in this project
            preact_noise=cfg.preact_noise,
            postact_noise=cfg.postact_noise
        )

        # Create the readout layer
        readout_layer = nnl.LinearLayer(input_dim=cfg.hidden_dim, output_dim=cfg.output_dim)

        # Initialize the parent RNNLayer with the constructed layers
        super().__init__(
            hidden_layers=[hidden_layer],
            readout_layer=readout_layer
        )
