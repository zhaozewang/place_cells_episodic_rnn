import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from trial_manager import TrialManager


class Trainer:
    """
    Trainer class to train the model
    """
    def __init__(
        self, 
        model: torch.nn,
        optimizer: torch.optim,
        loss: torch.nn,
        trial_manager: TrialManager
    ):
        """
        Initialize the Trainer

        Parameters:
            - model (torch.nn): The model to train
            - optimizer (torch.optim): The optimizer to use
            - loss (torch.nn): The loss function to use
            - trial_manager (TrialManager): The trial manager to use
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.trial_manager = trial_manager

    def train(self):
        """
        Train the model
        """
        self.model.train()
        self.trial_manager.new_trial()

        # Use tqdm with the trial iterator itself
        pbar = tqdm(self.trial_manager.trial, desc="AutoEncoding Trajectory", ncols=100, leave=True, total=self.trial_manager.trial.n_updates)
        
        for step, res in enumerate(pbar):
            self.optimizer.zero_grad()
            output, states = self.model(res)
            
            # Calculate the loss
            total_loss, loss_dict = self.loss({
                "pattern_completion": {
                    "pred": output,
                    "target": res,
                },
                "firing_rate": {
                    "state": states[0]
                }
            })
            
            total_loss.backward()
            self.optimizer.step()
            
            # Update the progress bar description with the current loss values
            loss_list = ', '.join([f"{v.item():.4f}" for v in loss_dict.values()])
            pbar.set_description(f"Step {step+1}/{self.trial_manager.trial.n_updates} | Loss: {loss_list}")
